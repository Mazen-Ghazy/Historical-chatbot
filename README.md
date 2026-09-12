# Historical Chatbot

An AI-powered historical chatbot built using **Retrieval-Augmented Generation (RAG)**.

The chatbot retrieves relevant information from historical PDF documents and uses **Google Gemini** to generate context-based answers.

## Features

- PDF-based question answering
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- ChromaDB vector database
- Google Gemini LLM
- Hugging Face embeddings
- Flask web application
- Responsive chatbot interface

## Technologies

- Python
- Flask
- LangChain
- Google Gemini
- ChromaDB
- Hugging Face
- PyPDF
- HTML / CSS / JavaScript

## Project Structure

```text
Historical-chatbot/
│
├── app.py                 # Flask application and RAG pipeline
├── store.py               # Creates the ChromaDB vector database
├── requirements.txt       # Project dependencies
│
├── src/
│   ├── helper.py          # PDF loading, text splitting and embeddings
│   └── prompt.py          # System prompt
│
├── templates/
│   └── chat.html          # Chatbot interface
│
├── static/
│   └── style.css          # Frontend styling
│
├── data/                  # Historical PDF documents
└── vectordb/              # ChromaDB vector database