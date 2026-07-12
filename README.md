# Research Paper Brain RAG

A simple Retrieval-Augmented Generation (RAG) project for processing research papers in PDF format, chunking them into smaller pieces, generating embeddings, storing them in a vector database, and answering questions using retrieved context.

## Overview

This project demonstrates a basic end-to-end RAG workflow:

1. Load PDF documents from the data folder.
2. Split the documents into meaningful chunks.
3. Generate embeddings for each chunk.
4. Store the embeddings in ChromaDB.
5. Retrieve relevant chunks for a user query.
6. Use an LLM to generate an answer from the retrieved context.

## Project Structure


Research-Paper-Brain-RAG/
├── app.py                 # Main entry point for the project
├── data/
│   └── docs/              # PDF files used for the RAG pipeline
├── notebook/              # Jupyter notebooks for experimentation
├── src/
│   ├── data_loader.py     # PDF loading and document chunking
│   ├── embedding.py       # Embedding generation using SentenceTransformers
│   ├── vectorstore.py    # ChromaDB vector store and retriever logic
│   └── search.py         # Simple RAG search and answer generation
├── requirement.txt       # Python dependencies
└── README.md             # Project documentation
```

## Requirements

Make sure you have Python 3.9+ installed.

Install the required packages:

```bash
pip install -r requirement.txt
```

## Environment Setup

Create a `.env` file in the project root and add your API key if you are using Groq:

```env
GROQ_API_KEY=your_api_key_here
```

## Running the Project

Run the main application:

```bash
python app.py
```

This will:
- load the PDFs from the data folder,
- split them into chunks,
- generate embeddings,
- store them in the vector database,
- and run a sample query.

## Notes

- The project uses ChromaDB for vector storage.
- The embedding model is loaded using SentenceTransformers.
- The retrieval and answer generation flow is implemented in the `src` folder.

## Future Improvements

- Add a web interface.
- Support more document formats.
- Improve chunking and retrieval quality.
- Add evaluation and testing.
