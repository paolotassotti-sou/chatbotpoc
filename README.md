# ChatbotPOC

A simple chatbot written in Python that answers questions using a local knowledge base of documents and a generative language model.  
It is composed of two distinct phases:
 - **Knowledge-Building**: It embeds all the documents inside a given directory. These embedding are then stored into a local persistent vector database along with metadata.
 - **Querying**: The querying phase implements a standard LLM pipeline to provide context-aware answers:
   * Retrieves embeddings from the database
   * Uses a Maximum Marginal Relevant strategy to improve answer diversity
   * Reranks retrieved documents based on the input query
   * Builds a prompt including a truncated context and chat history (only questions)

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

```
chatbotpoc/
├── embeddings/
│   ├── index.faiss
│   └── index.pkl
├── knowledge/
├── injest.py
├── knowledge.py
├── query.py
├── requirements.txt
└── README.md
```
