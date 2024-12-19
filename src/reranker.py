from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)
import torch
import requests
from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
    

def jina_reranker(query: str, documents: list[str], topk: int) -> list[str]:
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "top_n": topk,
        "documents": documents
    }

    topk = min(topk, len(documents))
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP issues
        response =  response.json()["results"]
        response = sorted(response, key=lambda x: x['relevance_score'], reverse=True)
        result = [item['document']['text'] for item in response]
        logger.success(f"reranked and extracted - {len(result)} documents")
        return result[:topk]

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred: {e}")
        return documents[:topk]
