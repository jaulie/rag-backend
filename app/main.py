import os
from fastapi import FastAPI, File, UploadFile
from mistralai import Mistral
from app.chunking import chunk_sentences, SemanticChunker
from app.read_pdf import read_pdf, save_file
from app.query_processing import is_retrieval_query
from app.vector_db import VectorDB
from app.embedding import MistralEmbedder

app = FastAPI()
embedder = MistralEmbedder()
chunker = SemanticChunker(embedder)
db = VectorDB(1024)

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# TODO: error handling for non-pdf file formats
# TODO: error handling with file sizes (large quantities of PDFs or long PDFs)
@app.post("/upload/")
async def read_pdf(files: list[UploadFile] = File(..., description="Upload one or more PDF files")):
    for file in files:
        content = await file.read()
        pdf_path = f"/tmp/{file.filename}"
        save_file(content, pdf_path)

        # Get the text from the pdf and chunk into sentences
        full_text = read_pdf(pdf_path)
        sentences = chunk_sentences(full_text)

        # Clean the path
        os.remove(pdf_path)

        # Semantic Chunking
        chunks = chunker.chunk(sentences)

        # Store in our VectorDB
        db.add_vectors(chunks)

@app.post("/query/")
async def query_rag(payload: QueryRequest):

    # If query doesn't require retrieval, answer directly
    if not is_retrieval_query(payload.query, client):
        # Directly answer with the LLM
        response = client.chat(prompt=payload.query)
        return {"answer": response, "source": "LLM only"}
    
    query_embedding = embedder.embed_texts([payload.query])[0]
    retrieved = db.search(query_embedding, top_k=5)

    context = "\n".join([item['text'] for item in retrieved])
    response = client.chat(prompt=f"{context}\n\nUser: {payload.query}")
    return {"answer": response, "source": "LLM with context"}