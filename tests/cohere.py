import os
import requests
from typing import List


COHERE_EMBED_URL = "https://api.cohere.ai/v1/embed"


def embed_texts_via_http(
    texts: List[str],
    model: str = "multilingual-22-12",
    timeout: float = 10.0
) -> List[List[float]]:

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    if COHERE_API_KEY is None:
        raise RuntimeError("Set COHERE_API_KEY in your environment")

    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "texts": texts
    }

    resp = requests.post(COHERE_EMBED_URL, headers=headers,
                         json=payload, timeout=timeout)
    resp.raise_for_status()

    body = resp.json()
    return body["embeddings"]


if __name__ == "__main__":
    # os.environ["COHERE_API_KEY"] = "<insert_api_key_here>"
    samples = ["Hello world", "How are you?"]
    vectors = embed_texts_via_http(samples)
    for text, vec in zip(samples, vectors):
        print(text, "->", vec[:5], "...")
