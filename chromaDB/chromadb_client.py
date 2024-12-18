import os
import chromadb
from chromadb.config import Settings
import numpy as np
import time
import openai
from dotenv import load_dotenv
import torch
from typing import List
import re
from pymilvus.model import sparse
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from args import get_args
import json

args = get_args()


class ChromaRetriever:
    def __init__(self, host: str = 'localhost', port: int = 5001, model_name: str = "text-embedding-ada-002"):

        self.client = chromadb.HttpClient(host=host, port=port)
        self.embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def get_collection(self, filename: str = 'documentembeddings') -> chromadb.Collection:

        collection = self.client.get_or_create_collection(
            name=filename, embedding_function=self.embedding_function
        )
        return collection

    def get_collection_name(file_name: str) -> str:
        collection_name = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
        collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
        collection_name = collection_name.strip("_-")

        if len(collection_name) > 63:
            collection_name = collection_name[:61] + "A"

        return collection_name

    @staticmethod
    def _clean_text_list(text_list: List[str]) -> List[str]:

        cleaned_list = []
        for text in text_list:
            # Remove HTML tags using regex
            text_cleaned = re.sub(r'<[^>]*>', '', text)
            # Replace \xa0 with a space
            text_cleaned = text_cleaned.replace('\xa0', ' ').replace('\n', ' ')
            # Replace multiple spaces with a single space
            text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
            cleaned_list.append(text_cleaned)
        return cleaned_list
    
    def get_num_chunks(self, collectino_name: str) -> int:
        collection = self.get_collection(collectino_name)
        return len(collection.get())

    def retrieve_documents(self, collection: chromadb.Collection, query: str, n_results: int = args.topk) -> List[str]:

        results = collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0]

    def adv_retrieve_documents(self, query: str, collection_name: str, n_results: int = args.topk) -> List[str]:
        """
        Retrieves top n documents from the Chroma collection based on the fused ranking of dense (OpenAI) and sparse (Splade) embeddings.
        """
        # Get Chroma collection
        collection = self.get_collection(collection_name)

        # Initialize Splade embedding function
        splade_ef = sparse.SpladeEmbeddingFunction(
            model_name="naver/splade-cocondenser-ensembledistil",
            device="cuda" if torch.cuda.is_available() else "cpu",
            # trust_remote_code=True
        )

        # make query embedding
        query_embedding_dense = openai.embeddings.create(
            input=query, model="text-embedding-ada-002").data[0].embedding
        query_embedding_dense = torch.tensor(query_embedding_dense)

        query_embedding_splade = splade_ef.encode_documents(
            [query])  # Ensure it's a list
        query_embedding_splade = query_embedding_splade.toarray()
        query_embedding_splade = query_embedding_splade.tolist()[0]
        query_embedding_splade = torch.tensor(query_embedding_splade)

        collection_data = collection.get(
            include=["embeddings", "documents", "metadatas"])
        
        # print (collection_data)
        
        n_results = min (n_results, len(collection_data['documents']))

        # document embedding
        # Dense embeddings from ChromaDB
        document_embeddings = np.array(collection_data['embeddings'])
        document_embeddings = torch.tensor(document_embeddings)

        # Corresponding document texts
        document_texts = collection_data['documents']

        # Corresponding document metadata (including Splade embeddings)
        document_metadatas = collection_data['metadatas']
        document_splade_embeddings = [json.loads(
            metadata.get("splade")) for metadata in document_metadatas]
        document_splade_embeddings = torch.tensor(document_splade_embeddings)

        print (document_splade_embeddings.shape)

        # Step 5: Compute Similarities
        dense_similarities = torch.nn.functional.cosine_similarity(
            query_embedding_dense, document_embeddings)

        # Sparse similarity (dot product for TF-IDF-style embeddings)
        sparse_similarities = torch.tensor([
            torch.dot(query_embedding_splade, doc_sparse) for doc_sparse in document_splade_embeddings
        ])

        # Step 6: Rank Fusion
        # Convert similarities to ranks (lower rank is better)
        # Negative for descending rank
        dense_ranks = rankdata(-dense_similarities.numpy(), method="min")
        sparse_ranks = rankdata(-sparse_similarities.numpy(), method="min")

        # Fuse ranks (using reciprocal rank fusion as an example)
        fused_ranks = 1 / dense_ranks + 1 / sparse_ranks

        # Step 7: Sort documents by fused ranks
        # Negative for descending order
        sorted_indices = np.argsort(-fused_ranks)
        top_indices = sorted_indices[:n_results]  # Top n results

        # Step 8: Retrieve top n documents
        top_documents = [document_texts[idx] for idx in top_indices]

        logger.debug(f"Retrieved {len(top_documents)} documents based on rank fusion.")
        return self._clean_text_list(top_documents)

    def return_final_retrieve_docs(self, query: str, collection_name: str, n_results: int = args.topk) -> List[str]:

        collection = self.get_collection(collection_name)
        n_results = min(n_results, len(collection.get()['documents']))
        retrieved_docs = self.retrieve_documents(
            collection=collection, query=query, n_results=n_results)
        logger.info(f"Retrieved {len(retrieved_docs)} documents based on rank fusion.")
        return self._clean_text_list(retrieved_docs)
    
    def get_document_context(self, collection_name: str) -> str:
        collection = self.get_collection(collection_name)
        collection_data = collection.get(
            include=["documents"])
        # Corresponding document texts
        document_texts = collection_data['documents']
        document_texts = self._clean_text_list(document_texts)
        logger.info(f"extracted document context")
        return " ".join(document_texts)


def main():

    def get_collection_name(file_name: str) -> str:
        collection_name = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
        collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
        collection_name = collection_name.strip("_-")

        if len(collection_name) > 63:
            collection_name = collection_name[:61] + "A"
        return collection_name

    query = "Write about Licenses and Use of the Chase Logos and Trademarks, as mentioned in the document?"

    chroma_retriever = ChromaRetriever(port=5001)
    start_time = time.time()

    # pdf_path = '/home/laksh-mendpara/Downloads/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf'
    pdf_path = "./uploaded_files/Nisarg_report.pdf"
    # pdf_path = "/home/laksh-mendpara/Downloads/BB84_Advanced Lab Experiment_2006.pdf"
    pdf_name = pdf_path.split('/')[-1]
    collection_name = get_collection_name(pdf_name)

    # results = chroma_retriever.adv_retrieve_documents(
    #     query=query,
    #     collection_name=collection_name,
    #     n_results=5
    # )

    # results = chroma_retriever.return_final_retrieve_docs(
    #     query=query,
    #     collection_name=collection_name,
    #     n_results=5
    # )

    results = chroma_retriever.adv_retrieve_documents(
        query=query,
        collection_name=collection_name,
        n_results=5
    )

    end_time = time.time()
    logger.debug(f"Retrieved documents: {results} ..... with number of documents: {len(results)}")
    logger.debug(f"Retrieval took: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
    # chroma_retriever = ChromaRetriever(port=5001)
    # def get_collection_name(file_name: str) -> str:
    #     collection_name = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
    #     collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
    #     collection_name = collection_name.strip("_-")

    #     if len(collection_name) > 63:
    #         collection_name = collection_name[:61] + "A"
    #     return collection_name

    # pdf_path = '/home/laksh-mendpara/Downloads/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf'
    # pdf_name = pdf_path.split('/')[-1]
    # collection_name = get_collection_name(pdf_name)
    # doc = chroma_retriever.get_document_context(collection_name)
    # print(doc)
