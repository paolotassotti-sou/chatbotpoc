# ChatbotPOC

A simple chatbot written in Python.
This is a basic POC that answers questions using a local knowledge base of documents and a generative language model.  
It unfolds into two phases: knowledge-building and querying.
In the first phase we build our knowledge base by embedding all the documents inside a given directory.
These embedding are then stored into a local persistent vector database along with metadata.
The querying phase implements a standard LLM pipeline to provide context-aware answers:
 - retrieve embeddings from the database
 - use a Maximum Marginal Relevant strategy to improve answer diversity
 - rerank retrieved documents based on the input query
 - build a prompt including a truncated context and chat history (only questions)

---

## Features

- Answers questions based on a local knowledge base.
- Easily extendable to ingest additional documents (text, PDFs, etc.).

---

## Prerequisites

- Python 3.10+
- Libraries:
  ```bash
  pip install torch transformers sentence-transformers langchain langchain-community faiss-cpu numpy nltk beautifulsoup4 lxml sentence-transformers



## Installation
git clone https://github.com/paolotassotti-sou/chatbotpoc.git
cd chatbotpoc


## (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows


## Install dependencies
pip install -r requirements.txt


## Project Structure

|-- ./pdf_to_text.py

|-- ./injest.py

|-- ./html_to_text.py

|-- ./LICENSE

|-- ./knowledge

|-- ./embeddings

|-- ./query.py

|-- ./knowledge.py

|-- ./requirements.txt

