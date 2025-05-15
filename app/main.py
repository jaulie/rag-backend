import os
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from chunking import chunk_sentences, SemanticChunker
from pdf_extraction import extract_text, save_file
from query_processing import is_retrieval_query, run, run_with_context
from vector_db import VectorDB
from embedding import MistralEmbedder

app = FastAPI()
embedder = MistralEmbedder()
chunker = SemanticChunker(embedder)
db = VectorDB(1024, embedder=embedder)

# TODO: error handling for non-pdf file formats
# TODO: error handling with file sizes (large quantities of PDFs or long PDFs)
@app.post("/upload/")
async def read_pdf(files: list[UploadFile] = File(..., description="Upload one or more PDF files")):
    """This is our pdf extraction endpoint. The pdf is semantically chunked and stored in a vector DB."""
    responses = []
    for file in files:
        content = await file.read()
        pdf_path = f"/tmp/{file.filename}"
        save_file(content, pdf_path)

        # Get the text from the pdf and chunk into sentences
        full_text = extract_text(pdf_path)
        sentences = chunk_sentences(full_text)

        # Clean the path
        os.remove(pdf_path)

        # Semantic Chunking
        chunks = chunker.chunk(sentences)

        # Store in our VectorDB
        db.add_chunks(chunks)
        responses.append({"filename": file.filename, "chunks_added": len(chunks)})
    return {"status": "success", "details": responses}

class QueryRequest(BaseModel):
    """When inputting the query, user can specify their desired mode of search"""
    query: str
    search_type: str = "semantic" # "semantic" is default, but "keyword" and "hybrid" are also valid

@app.post("/query/")
async def query_rag(payload: QueryRequest):
    """This is our RAG endpoint. There are 3 options for search, with potential to add
    arguments in order to tune hybrid search (like a UI slider)"""

    # If query doesn't require retrieval, answer directly
    if not is_retrieval_query(payload.query):
        response = run(payload.query)
        return {"answer": response, "source": "LLM only"}
    
    # Else, we want to embed the query
    query_embedding = embedder.embed_texts([payload.query])[0]
    if payload.search_type == "keyword":
        retrieved = db.keyword_search(payload.query, num_results=3)  
        print(retrieved)
        context = "\n".join([item['text'] for item in retrieved])
        response = run_with_context(payload.query, context)
        return {"answer": response, "source": "Keyword search with LLM"}
    elif payload.search_type == "semantic":
        retrieved = db.search(query_embedding, num_results=3)
        print(retrieved[0])
        context = "\n".join([item['text'] for item in retrieved])
        response = run_with_context(payload.query, context)
        return {"answer": response, "source": "Semantic search with LLM"}
    elif payload.search_type == "hybrid":
        retrieved = db.hybrid_search(payload.query, query_embedding, num_results=3)
        context = "\n".join([item['text'] for item in retrieved])
        response = run_with_context(payload.query, context)
        return {"answer": response, "source": "Hybrid search with LLM"}
    else:
        return {"error": "Invalid search_type. Please choose 'semantic', 'keyword', or 'hybrid'."}
