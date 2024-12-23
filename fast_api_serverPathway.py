import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
#from PyPDF2 import PdfReader
import os
from loguru import logger
from PathwayVectorStore.runVectorStore import run_vector_store
import time
import threading
from mainPathway import pipeline

app = FastAPI()

# Ensure the directory for uploaded files exists
os.makedirs("./uploaded_files", exist_ok=True)


# FastAPI endpoint to upload file
@app.post("/upload")
async def save_file(file: UploadFile, drive_link: str):
    global collection_name_global
    collection_name_global = ""
    logger.success(f"Drive Link received: {drive_link}")
    temp_file_path = os.path.join("./uploaded_files", file.filename)

    object_id = drive_link.split("/")[-1]

    logger.success(f"Object ID: {object_id}")
    logger.success(f"File Path: {temp_file_path}")

    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    run_vector_store(
        credential_path=temp_file_path,
        object_id=object_id
    )

    return ""

# FastAPI endpoint to process text
@app.get("/process")
async def process_text(prompt: str,collection_name_global):
    if collection_name_global is None:
        return {"error": "No collection name available. Please upload a file first."}
    logger.info(f"process call name - {collection_name_global}")
    final_response = pipeline(
        query=prompt,
        topk=8,
        reranker=True,
        method="cr",
        agent_type="dynamic",
        use_reflection=False,
        n_reflection=0,
        use_router=True
    )
    return {"response_markdown":final_response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
