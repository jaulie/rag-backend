# RAG Backend with FastAPI and PDF Ingestion #

This project implements a Retrieval-Augmented Generation (RAG) backend using FastAPI that allows uploading one or more PDF files, parsing them with pymupdf4llm, chunking their contents, storing them in a proprietary vector store, and answering queries with optional semantic search.


## Features ##
- Upload one or more PDF files via API
- PDF content is chunked, embedded, and stored in a vector database    
- Smart query pre-processing to decide if retrieval is needed
- Retrieve and return relevant document chunks for the query

## Project Structure ##
```
rag-backend/
├── app/
│   ├── chunking.py         # Preliminary chunking & semantic chunking
│   ├── embedding.py        # Embedding model
│   ├── main.py             # Endpoints for FastAPI
│   ├── query_processing.py # Query handling functions           
│   ├── read_pdf.py         # PyMuPDF text extraction functions
│   └── vector_db.py        # VectorDB with FAISS search
├── data/                   # Optional storage (e.g., temp PDFs)
├── requirements.txt
└── README.md
```

## Requirements ##

## File Upload Endpoint ##
To upload files run
```uvicorn main:app --reload```
inside the app directory. Then, [open the UI](http://127.0.0.1:8000/docs).
Click POST and the choose file(s) to upload.

## PDF Text Extraction and Chunking ##
PDFs are extracted using PyMuPDF. Chunking is done using semantic chunking with the difference threshold set by 
1 standard deviation in the data. This is the most versatile chunking strategy given the use case.

## Query Processing ##
The query processor determines:
- If the question actually requires retrieval    
- If search should be semantic or keyword-based    
- Whether fallback logic is needed (e.g., no matches found)

## Vector Store Integration ##
Documents are embedded and stored in a proprietary vector store.
Search is implemented using FAISS.

## Embeddings ##
We use `mistral-embed` as the embedding model.