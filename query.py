#!/usr/bin/python

# Copyright (c) 2025, Paolo Tassotti <paolo.tassotti@gmail.com>
# GNU General Public License v3.0+ (see LICENSES/GPL-3.0-or-later.txt or https://www.gnu.org/licenses/gpl-3.0.txt)


import os
import sys
import time


print('Importing AI packages...', end='')
sys.stdout.flush()

import nltk
import torch
import pickle
import numpy as np

#nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores.utils import maximal_marginal_relevance


from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings




print("done.")


# Set device to MPS (Apple GPU) if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# Loading the knowledge base
print('Loading the knowledge base...', end='')
sys.stdout.flush()

model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(device)


# initialize cross encoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


with open("metadata.pkl", "rb") as f:
  metadata = pickle.load(f)


# wrap metadata in Document objects
docs = [Document(page_content=doc["text"], metadata=doc) for doc in metadata]

# Wrap embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Create FAISS vectorstore from documents (builds new index)
vectorstore = FAISS.from_documents(docs, embeddings)

# Get retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

print("done.")


# filter retrieved documents based on cosine similarity with the query.
def filter_by_cosine_similarity(query_text, retrieved_docs, model, threshold=0.65):

  # Encode query
  query_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]

  filtered_docs = []
  for doc in retrieved_docs:
    doc_emb = model.encode([doc.page_content], convert_to_numpy=True, normalize_embeddings=True)[0]

    # cosine similarity (since already normalized: dot product = cosine similarity)
    sim = np.dot(query_emb, doc_emb)

    if sim >= threshold:
      filtered_docs.append(doc)

    return filtered_docs


def mmr_relevant_docs(text, top_k=10):

    # Use the retriever from vectorstore
    retrieved_docs = retriever.invoke(text)

    # Apply cosine similarity filter
    retrieved_docs = filter_by_cosine_similarity(text, retrieved_docs, model, threshold=0.65)

    # compute embeddings for MMR
    candidate_embeddings = np.array([
        model.encode([doc.page_content], convert_to_numpy=True)[0]
        for doc in retrieved_docs
    ])

    # compute query embedding
    query_embedding = model.encode([text], convert_to_numpy=True)[0]
    query_embedding /= np.linalg.norm(query_embedding)  # normalize

    # apply MMR
    selected_indices = maximal_marginal_relevance(
        query_embedding=query_embedding,
        embedding_list=candidate_embeddings,
        k=min(top_k, len(retrieved_docs)),
        lambda_mult=0.5
    )

    retrieved_docs = [retrieved_docs[i] for i in selected_indices]

    return retrieved_docs, candidate_embeddings



# rerank a list of retrieved_docs (dicts with 'text') based on query relevance
# returns the top_n most relevant docs
def rerank(query_text, retrieved_docs, top_n=5):

    if not retrieved_docs:
        return []

    # Prepare pairs (query, doc_text) for the cross-encoder
    pairs = [(query_text, doc.page_content) for doc in retrieved_docs]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort by score descending
    ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)

    # Return top_n documents
    return [doc for _, doc in ranked[:top_n]]


def generate_answer(question, gen_model, tokenizer, history_string=None, top_k=20, max_length=300, device=None, use_fp16=False):

    max_input_tokens = 512  # Flan-T5 max input

    # Detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if history_string is not None:
      retrieval_query = f"{history_string}\nUser: {question}"
    else:
      retrieval_query = f"User: {question}"

    retrieved_docs, embeddings = mmr_relevant_docs(retrieval_query, top_k=10)

    retrieved_docs = rerank(retrieval_query, retrieved_docs, top_n=5)

    # Build context from reranked Document objects
    context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])


    # truncating at sentence boundaries
    sentences = sent_tokenize(context)
    truncated_tokens = []
    token_count = 0

    for s in sentences:
      s_tokens = tokenizer.encode(s + '.', add_special_tokens=False)
      if token_count + len(s_tokens) > max_input_tokens:
        break
      truncated_tokens.extend(s_tokens)
      token_count += len(s_tokens)

    truncated_context = tokenizer.decode(truncated_tokens)


    # Build prompt for generation model
    prompt = f"""You are an expert explaining this topic. Answer the CURRENT question only, using the context. 
    Do NOT repeat answers from previous turns.


    Use the following context to answer the question. 
    Ignore previous answers:
    {truncated_context}

    Conversation History:
    {history_string}

    Current Question: {question}

    Instructions:
    Answer concisely
    """


    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,       # truncate to max_length
        max_length=max_input_tokens
    ).to(device)

    # Move model to the chosen device
    gen_model.to(device)

    # Optional: convert model to half precision
    if use_fp16:
        gen_model.half()

    # Generate output sequence
    outputs = gen_model.generate(
        **inputs,
        max_length=max_length,
        min_length=80,        # force at least some length
        num_beams=4,          # better coverage
        no_repeat_ngram_size=3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# Load the genarational model and the tokenizer
print("Loading the generational model...", end=' ')

model_name = "google/flan-t5-large"  # You can switch to bart, t5 variants, etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("done.")


# Optional: enable half-precision for speed (works if model supports it)
#use_fp16 = False

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


history_max_len = 3
chat_history = []
while True:

  print("Question: ", end='')
  question = input()

  if question.lower() == "quit" or question.lower() == "exit" or question.lower() == "break":
    break

  start_time = time.time()  # Start timer

  # consolidate chat history (only questions) in a string
  history_string = "\n".join([msg['content'] for msg in chat_history if msg['role'] == 'user'][-history_max_len:])

  answer = generate_answer(question, gen_model, tokenizer, history_string=history_string, device=device)

  end_time = time.time()  # End timer
  elapsed_time = end_time - start_time

  print(f"Answer: {answer}")
  print("")
  print(f"Time taken to generate the answer: {elapsed_time:.4f} seconds")

  # update chat history
  chat_history.append({"role": "user",      "content": question})
  chat_history.append({"role": "assistant", "content": answer})

  # handle history as a queue
  if len(chat_history) > history_max_len:
    chat_history = chat_history[2:] # remove oldest two messages (user and assistant)

