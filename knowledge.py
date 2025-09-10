#!/usr/bin/python

# Copyright (c) 2025, Paolo Tassotti <paolo.tassotti@gmail.com>
# GNU General Public License v3.0+ (see LICENSES/GPL-3.0-or-later.txt or https://www.gnu.org/licenses/gpl-3.0.txt)

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings


import injest
import os

# constants
EMBED_DIR = "embeddings/"
KNOWLEDGE_DIR = "knowledge/"

# Ensure output dir exists
os.makedirs(EMBED_DIR, exist_ok=True)


# Load the embedding model (via LangChain wrapper)
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Load documents from your ingestion pipeline
processed_docs = injest.process_knowledge_dir(KNOWLEDGE_DIR)

# initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=500,     # ~500 tokens per chunk
  chunk_overlap=50,
  separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " "]
)


# Convert your ingested docs into LangChain Documents
docs = []
for doc in processed_docs:
  body_text = doc.get("body_text", "").strip()
  if not body_text:
    continue

  title = doc.get("title", "Untitled")
  source_file = doc.get("source_file", None)

  print(f"Processing {title}...")

  # Split text
  chunks = text_splitter.split_text(body_text)

  # Clean up chunks to avoid double dots and ensure proper punctuation
  cleaned_chunks = []
  for c in chunks:
    c = c.strip()         # remove leading/trailing whitespace
    c = c.rstrip(".")     # remove trailing periods
    if c:                 # skip empty chunks
      cleaned_chunks.append(c + ".")  # add exactly one period

  chunks = cleaned_chunks


 # chunks = [c.lstrip(". ").strip() for c in chunks]  # cleanup

  # Wrap chunks in Document objects with metadata
  for chunk in chunks:
    docs.append(
      Document(
        page_content=chunk,
        metadata={
          "source_title": title,
          "source_file": source_file
        }
      )
    )

# Create FAISS vectorstore
print("Building FAISS index...")
vectorstore = FAISS.from_documents(docs, embeddings)

# Save to disk (this creates index.faiss + index.pkl in EMBED_DIR)
vectorstore.save_local(EMBED_DIR)

print(f"Knowledge base saved to {EMBED_DIR}")

