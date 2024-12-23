import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PyPDF2 import PdfReader
import os
from loguru import logger
from chromaDB.chromadb_server import CHROMADB
from main import pipeline

app = FastAPI()
chroma_server = CHROMADB(port=5001)

# Ensure the directory for uploaded files exists
os.makedirs("uploaded_files", exist_ok=True)


# FastAPI endpoint to upload file
@app.post("/upload")
async def save_file(file: UploadFile):
    global collection_name_global

    temp_file_path = f"uploaded_files/{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    # Read the PDF using PyPDF2
    pdf_reader = PdfReader(temp_file_path)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    # Use CHROMADB to get collection name and embed PDF
    collection_name = chroma_server.get_collection_name(file.filename)
    embed = CHROMADB()
    embed.embed_pdf(pdf_path=temp_file_path, method="cr")
    logger.info(collection_name)
    return collection_name

# FastAPI endpoint to process text
@app.get("/process")
async def process_text(prompt: str,collection_name_global):
    if collection_name_global is None:
        return {"error": "No collection name available. Please upload a file first."}
    logger.info(f"process coll name - {collection_name_global}")
    final_response = pipeline(
        query=prompt,
        collection_name=collection_name_global,
        topk=6,
        method="cr",
        use_reflection=False,
        n_reflection=1,
        agent_type="dynamic"
    )
    return {"response_markdown": final_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
