# RAG Backend with FastAPI and PDF Ingestion #

This project implements a Retrieval-Augmented Generation (RAG) backend using FastAPI that allows uploading one or more PDF files, parsing them with pymupdf, chunking their contents, storing them in a proprietary vector store, and answering queries with either semantic search, keyword search, or a hybrid search option.


## Features ##
- Upload one or more PDF files via API
- PDF content is chunked, embedded, and stored in a vector database    
- Smart query pre-processing to decide if retrieval is needed
- Retrieve and return relevant document chunks for the query using 3 different search strategies
- Input relevant document chunks as context using MistralAI

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
Requires a Mistral API key.
For keyword search, downloading stopwords is required:      
`>>> nltk.download('stopwords')`        
This enables stopword removal before keyword matching. 
For packages, see requirements file.

## File Upload Endpoint ##
To upload files run
```uvicorn main:app --reload```
inside the app/ directory. Then, [open the UI](http://127.0.0.1:8000/docs).
Click , then 'Try it out', then choose file(s) to upload.

## PDF Text Extraction and Chunking ##
PDFs are extracted using PyMuPDF. Chunking is done using semantic chunking with the difference threshold set by 
1 standard deviation in the data. This is the most versatile chunking strategy given the non-specific domains of the pdf inputs.

## Vector Store Integration ##
Documents are embedded and stored in a proprietary vector store.
Semantic search is implemented using FAISS. Keyword search simply removes stop words and
determines similarity based on keyword count.   
Hybrid search combines both approaches 50/50. In future iterations, the hybrid search function could include a slider
so that the user can weigh which search option they prefer to be dominant.

## Query Endpoint ##
User enters their query. The user can optionally specify the search type (semantic, keyword, or hybrid).

## Query Processing ##
The query processor determines:
- If the question actually requires retrieval. This is done by invoking the model.
- If search should be semantic, keyword-based, or hybrid. This is done based on input arguments.
Fallback logic is needed, since search is guaranteed to return a result, but not guaranteed to return
meaningfully similar texts. The number of texts returned would ideally be variable and dependent on a 
user-defined similarity threshold. 

## Embeddings ##
We use `mistral-embed` as the embedding model.

## Search ##
Enter 'semantic', 'keyword', or 'hybrid' for the three types of search.