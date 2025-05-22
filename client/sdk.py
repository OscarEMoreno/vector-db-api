import requests
from uuid import UUID
from typing import Any, Dict, List, Optional
from requests.exceptions import HTTPError, JSONDecodeError


class VectorDBClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 5):
        self.base: str = base_url.rstrip("/")
        self.timeout = timeout

    def create_library(self, name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('post', '/libraries', json={"name": name, "metadata": metadata})

    def get_library(self, lib_id: str) -> Dict[str, Any]:
        return self._request('get', f'/libraries/{lib_id}')

    def update_library(self, lib_id: str, name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('put', f'/libraries/{lib_id}', json={"name": name, "metadata": metadata})

    def delete_library(self, lib_id: str) -> None:
        self._request('delete', f'/libraries/{lib_id}')

    def create_document(
        self,
        lib_id: str,
        doc_id: str,
        title: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:

        return self._request(
            'post',
            f"/libraries/{lib_id}/documents",
            json={"id": str(doc_id), "title": title, "metadata": metadata},
        )

    def add_chunk(self, lib_id: str, doc_id: UUID, text: str,
                  embedding: List[float], metadata: Dict[str, Any]) -> Dict[str, Any]:
        return self._request('post', f'/libraries/{lib_id}/chunks',
                             json={"doc_id": str(doc_id), "text": text,
                                   "embedding": embedding, "metadata": metadata})

    def get_chunks(self, lib_id: str) -> List[Dict[str, Any]]:
        return self._request('get', f'/libraries/{lib_id}/chunks')

    def update_chunk(self, lib_id: str, chunk_id: UUID,
                     text: Optional[str] = None,
                     embedding: Optional[List[float]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if text is not None:
            payload['text'] = text
        if embedding is not None:
            payload['embedding'] = embedding
        if metadata is not None:
            payload['metadata'] = metadata
        return self._request('put', f'/libraries/{lib_id}/chunks/{chunk_id}', json=payload)

    def delete_chunk(self, lib_id: str, chunk_id: UUID) -> None:
        return self._request('delete', f'/libraries/{lib_id}/chunks/{chunk_id}')

    def search(
        self,
        lib_id: str,
        embedding: List[float],
        k: int = 1,
        algorithm: str = "kd",
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        body = {"embedding": embedding, "k": k, "algorithm": algorithm}
        if metadata_filter:
            body['metadata_filter'] = metadata_filter
        return self._request('post', f'/libraries/{lib_id}/search', json=body)['results']

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = self.base + path
        for attempt in range(3):
            resp = getattr(requests, method)(
                url, timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception as e:
                raise e

    def _request(self, method: str, path: str, json: Dict[str, Any] = None) -> Any:
        url = self.base + path
        resp = getattr(requests, method)(url, json=json, timeout=self.timeout)
        try:
            resp.raise_for_status()
        except HTTPError as e:
            raise
        try:
            return resp.json()
        except (ValueError, JSONDecodeError):
            return None
