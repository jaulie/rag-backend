import os
from fastapi import FastAPI, File, UploadFile
from app.chunking import chunk_sentences, SemanticChunker
from app.read_pdf import read_pdf, save_file
from app.vector_db import VectorDB

app = FastAPI()
chunker = SemanticChunker("mistral-embed")
db = VectorDB(1024)

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
        vectordb.add_chunks(chunks)

@app.post("/query/")
async def query_rag(payload: QueryRequest):
    query_embedding = embedder.embed_texts([payload.query])[0]
    retrieved = vectordb.search(query_embedding, top_k=5)

    context = "\n".join([item['text'] for item in retrieved])
    response = mistral_llm.chat(prompt=f"{context}\n\nUser: {payload.query}")
    return {"answer": response}