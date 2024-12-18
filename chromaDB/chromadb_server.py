import os
from time import time
from typing import List, Dict

import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import camelot
import torch
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus.model import sparse
from loguru import logger
from tqdm import tqdm
import json
import fitz  # PyMuPDF
import re
from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

DOCUMENT_CONTEXT_PROMPT = """
    <document>
    {doc_content}
    </document>
"""

CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole document
    
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
"""


class CHROMADB:
    def __init__(self, port: int=5001):
        self.client = chromadb.HttpClient(host="localhost", port=port)
        self.openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-ada-002",
            api_key=api_key
        )

    @staticmethod
    def _get_chunk_summary(doc: str, chunk: str) -> str:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a useful assistant"},
                {
                    "role": "user",
                    "content": (
                        DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)
                        + CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)
                    ),
                },
            ],
        )
        return completion.choices[0].message.content

    def custom_chunk_document_pdf(
        self,
        pdf_path: str,
        chunk_size=1000,
        overlap=200
    ) -> List[str]:

        table_chunks = []
        text_chunks = []
        # Extract tables using Camelot
        tables = camelot.read_pdf(pdf_path, pages='1-end', flavor='stream')
        for table in tables:
            df = table.df  # Get DataFrame of the table
            table_chunks.append(df.to_string(index=False))
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text_chunks.append(text.strip())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap)
        final_text_chunks = []
        for chunk in text_chunks:
            final_text_chunks.extend(splitter.split_text(chunk))

        return final_text_chunks + table_chunks

    def adv_custom_chunk_document_pdf(
        self,
        pdf_path: str,
        chunk_size=1000,
        overlap=200,
    ) -> List[str]:
        table_chunks = []
        text_chunks = []

        # Extract tables using Camelot
        tables = camelot.read_pdf(pdf_path, pages='1-end', flavor='stream')
        for table in tables:
            df = table.df  # Get DataFrame of the table
            table_chunks.append(f"<table>{df.to_string(index=False)}</table>")

        # Extract text from the PDF using PyMuPDF
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text_chunks.append(text.strip())

        # Split the text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        final_text_chunks = []

        # for chunk in tqdm(text_chunks, total=len(text_chunks), desc="Chunking and getting context for text"):
        #     page_chunks = splitter.split_text(chunk)

        #     for text_chunk in page_chunks:
        #         context = self._get_chunk_summary(chunk, text_chunk)
        #         final_chunk = (
        #             f"<chunk> {text_chunk} </chunk>\n\n\n" +
        #             f"<context>{context}</context>"
        #         )
        #         final_text_chunks.append(final_chunk)

        for chunk in tqdm(text_chunks, total=len(text_chunks), desc="Chunking and getting context for text"):
            page_chunks = splitter.split_text(chunk)

            with ThreadPoolExecutor() as executor:
                context = [executor.submit(self._get_chunk_summary, chunk, text_chunk) for text_chunk in page_chunks]
                
            final_chunk = [f"<chunk> {text_chunk} </chunk>\n\n\n" +
                    f"<context>{context}</context>" for text_chunk, context in zip(page_chunks, context)]
            # print (final_chunk)
            final_text_chunks.extend(final_chunk)

        logger.debug(f"Extracted {len(final_text_chunks)} text chunks from the PDF.")

        # Combine text chunks and table chunks
        return final_text_chunks + table_chunks

    def embed_and_store_chunks(self, doc_id: str, pdf_path: str, collection: chromadb.Collection):
        """
        Embeds document chunks and stores them in Chroma DB.

        Args:
            doc_id (str): Unique identifier for the document.
            pdf_path (str): Path to the PDF document.
            collection (chromadb.Collection): Collection to store document embeddings.
        """
        chunks = self.custom_chunk_document_pdf(pdf_path)
        for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc=f"Storing chunks for document {doc_id}"):
            # Use the OpenAI embedding function
            embedding = openai.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            ).data[0].embedding
            collection.add(
                ids=[f"{doc_id}_{idx}"],            # Unique ID for each chunk
                documents=[chunk],
                embeddings=[embedding]
            )

        logger.info(f"Stored {len(chunks)} chunks for document {doc_id}.")

    def adv_embed_and_store_chunks(self, doc_id: str, pdf_path: str,
                                   collection: chromadb.Collection, method: str = "cr") -> None:

        if method == "cr":
            # Step 1: Chunk the PDF document
            chunks = self.adv_custom_chunk_document_pdf(pdf_path)
        elif method == "hs":
            chunks = self.custom_chunk_document_pdf(pdf_path)

        # Step 2: Compute TF-IDF vectors
        splade_ef = sparse.SpladeEmbeddingFunction(
            model_name="naver/splade-cocondenser-ensembledistil",
            device= "cpu"
        )

        splade_docs_embeddings = splade_ef.encode_documents(chunks)
        splade_docs_embeddings = splade_docs_embeddings.toarray()
        splade_docs_embeddings = splade_docs_embeddings.tolist()

        # Step 3: Store chunks in Chroma DB
        # for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc=f"Storing chunks for document {doc_id}"):
        #     # Compute embedding using OpenAI's API
        #     embedding = openai.embeddings.create(
        #         input=chunk,
        #         model="text-embedding-ada-002"
        #     ).data[0].embedding
        #     # Store embedding and TF-IDF vector in ChromaDB
        #     splade_metadata = json.dumps(splade_docs_embeddings[idx])
        #     collection.add(
        #         ids=[f"{doc_id}_{idx}"],  # Unique ID for each chunk
        #         documents=[chunk],        # Original text chunk
        #         embeddings=[embedding],   # Embedding vector
        #         metadatas=[{
        #             "splade": splade_metadata  # Store TF-IDF as metadataz
        #         }]
        #     )

        def process_chunk(idx, chunk, doc_id, splade_docs_embeddings):
            """Function to process a single chunk."""
            # Compute embedding using OpenAI's API
            embedding = openai.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            ).data[0].embedding

            # Prepare metadata
            splade_metadata = json.dumps(splade_docs_embeddings[idx])

            # Return data to store in ChromaDB
            return {
                "id": f"{doc_id}_{idx}",
                "chunk": chunk,
                "embedding": embedding,
                "metadata": {"splade": splade_metadata}
            }

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor() as executor:
            # Submit tasks for all chunks
            futures = [
                executor.submit(process_chunk, idx, chunk, doc_id, splade_docs_embeddings)
                for idx, chunk in enumerate(chunks)
            ]

            # Collect and store results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Storing chunks for document {doc_id}"):
                try:
                    result = future.result()
                    # Store embedding and TF-IDF vector in ChromaDB
                    collection.add(
                        ids=[result["id"]],
                        documents=[result["chunk"]],
                        embeddings=[result["embedding"]],
                        metadatas=[result["metadata"]]
                    )
                except Exception as e:
                    print(f"Error processing chunk: {e}")


        logger.info(f"Stored {len(chunks)} chunks for document {doc_id} along SPADE embedding.")

    @staticmethod
    def get_collection_name(file_name: str) -> str:
        collection_name = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
        collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
        collection_name = collection_name.strip("_-")

        if len(collection_name) > 63:
            collection_name = collection_name[:61] + "A"

        return collection_name

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.openai_ef
        )
        return collection

    def embed_pdf(self, pdf_path: str, method: str = None):
        pdf_name = pdf_path.split('/')[-1]
        collection_name = self.get_collection_name(pdf_name)
        collection = self.get_collection(collection_name)

        if method == None:
            self.embed_and_store_chunks(
                doc_id=collection_name,
                pdf_path=pdf_path,
                collection=collection
            )
        elif method == "cr":
            self.adv_embed_and_store_chunks(
                doc_id=collection_name,
                pdf_path=pdf_path,
                collection=collection,
                method=method
            )
        elif method == "hs":
            self.adv_embed_and_store_chunks(
                doc_id=collection_name,
                pdf_path=pdf_path,
                collection=collection,
                method=method
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        logger.info(f"PDF {pdf_name} embedded and stored in collection {collection_name}.")

    def embed_dir(self, root: str, method: str = None):
        pdf_filepaths = [
            os.path.join(dirpath, file)
            for dirpath, _, files in os.walk(root)
            for file in files if file.lower().endswith(".pdf")
        ]

        for pdf_filepath in tqdm(pdf_filepaths, desc="Embedding PDFs"):
            self.embed_pdf(pdf_filepath, method=method)


def main():
    # Initialize Chroma DB through command chroma run --path /db_path
    chromadb_server = CHROMADB(5001)
    # pdf = '/home/laksh-mendpara/Downloads/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf'
    pdf = "./uploaded_files/Nisarg_report.pdf"
    # pdf = "/Users/nisarg/Desktop/Credit.pdf"
    # dir = '/home/laksh-mendpara/Downloads/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/'
    chromadb_server.embed_pdf(pdf_path=pdf, method="cr")
    # chromadb_server.embed_dir(root=dir, method="cr")


if __name__ == "__main__":
    main()
