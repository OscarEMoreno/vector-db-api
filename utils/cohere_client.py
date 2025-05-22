from typing import List, Optional
import os
import requests
import numpy as np
from dotenv import load_dotenv


load_dotenv()

class CohereClient:    
    EMBED_URL = "https://api.cohere.ai/v1/embed"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key must be provided or set in COHERE_API_KEY env var")
        self.model = "embed-english-v3.0"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            payload = {
                "model": self.model,
                "texts": [text],
                "input_type": "search_document"
            }
            
            response = requests.post(
                self.EMBED_URL,
                headers=self.headers,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            
            embeddings = response.json()["embeddings"]
            return np.array(embeddings[0], dtype=np.float32).tolist()
        except Exception as e:
            raise ValueError(f"Failed to generate embedding: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            payload = {
                "model": self.model,
                "texts": texts,
                "input_type": "search_document"
            }
            
            response = requests.post(
                self.EMBED_URL,
                headers=self.headers,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            
            embeddings = response.json()["embeddings"]
            return [np.array(emb, dtype=np.float32).tolist() 
                   for emb in embeddings]
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {str(e)}")

    @property
    def embedding_size(self) -> int:
        return 1024  # embed-english-v3.0 