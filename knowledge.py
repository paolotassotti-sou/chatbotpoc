#!/usr/bin/python

# Copyright (c) 2025, Paolo Tassotti <paolo.tassotti@gmail.com>
# GNU General Public License v3.0+ (see LICENSES/GPL-3.0-or-later.txt or https://www.gnu.org/licenses/gpl-3.0.txt)


from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

import numpy as np
import faiss
import pickle

import injest


# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
#model = SentenceTransformer('all-mpnet-base-v2')
#model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


# initialize vector database
dimension = 384  # embedding size for MiniLM-L6-v2
index = faiss.IndexFlatIP(dimension)
metadata = []


# extract text from all files in the knowledge base
knowledge_dir = './knowledge/'
processed_docs = injest.process_knowledge_dir(knowledge_dir)


# initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ~500 tokens per chunk
    chunk_overlap=50,      # small overlap for continuity
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " "]
)


# embed files
for doc in processed_docs:
  body_text = doc.get("body_text", "").strip()
  if not body_text:
    continue  # skip empty docs

  title = doc.get("title", "Untitled")
  source_file = doc.get("source_file", None)

  print("Embedding " + title)


  # split text into chunks
  print("split into chunks")
  chunks = text_splitter.split_text(body_text)

  # Clean leading dots/spaces in each chunk
  chunks = [c.lstrip(". ").strip() for c in chunks]


  # Create embeddings
  print("calculate embeddings")
  embeddings = model.encode(chunks, convert_to_numpy=True)
  print(embeddings.shape)  # (number_of_sentences, embedding_dimension)


  # cosine similarity
  # normalize embeddings to unit vectors
  embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


  # Add embeddings to the index
  start_pos = index.ntotal
  index.add(embeddings)
  print(f"Total vectors indexed: {index.ntotal}")

  # Add metadata for each chunk
  print("Adding metadata")
  for i, chunk in enumerate(chunks):
    metadata.append({
      "index": start_pos + i,
      "text": chunk,
      "source_title": title,
      "source_file": source_file
    })

  print("")



# Save FAISS index to file
print("Save embeddings data to disk")
faiss.write_index(index, "faiss_index.bin")

# Save metadata (texts + additional info) to a file with pickle
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

