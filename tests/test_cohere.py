import os
import requests
from typing import List
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from dotenv import load_dotenv


load_dotenv()
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


@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "embeddings": [np.random.randn(1024).tolist()]
    }
    return mock


def test_cohere_client(mock_response):
    from utils.cohere_client import CohereClient
    client = CohereClient(api_key="test-key")
    assert client.api_key == "test-key"
    assert client.model == "embed-english-v3.0"


    with patch('requests.post', return_value=mock_response) as mock_post:
        client = CohereClient(api_key="test-key")
        embedding = client.get_embedding("Hello, world!")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == COHERE_EMBED_URL
        assert kwargs['json']['texts'] == ["Hello, world!"]
        assert kwargs['headers']['Authorization'] == "Bearer test-key"
        assert isinstance(embedding, list)
        assert len(embedding) == client.embedding_size
        assert all(isinstance(x, float) for x in embedding)


if __name__ == "__main__":
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    samples = ["Hello world", "How are you?"]

    if not COHERE_API_KEY:
        exit(1)

    vectors = embed_texts_via_http(samples)
    for text, vec in zip(samples, vectors):
        print(text, "->", vec[:5], "...")
