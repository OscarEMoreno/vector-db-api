from uuid import uuid4, UUID
from typing import List, Dict, Any, Optional
import numpy as np

from domain.models import Library, Document, Chunk
from infrastructure.index.kdtree import KDTree
from infrastructure.index.balltree import BallTree
from infrastructure.index.linear import LinearIndex
from utils.pagination import paginate


class LibraryService:
    def __init__(self, repo):
        self.repo = repo

    def create_library(self, name: str, metadata: Dict[str, Any]) -> Library:
        lib = Library(id=uuid4(), name=name, documents=[], metadata=metadata)
        self.repo.add(lib)
        return lib

    def get_library(self, lib_id: str) -> Library:
        lib = self.repo.get(lib_id)
        if not lib:
            raise ValueError('Library not found')
        return lib

    def update_library(self, lib_id: str, name: str, metadata: Dict[str, Any]) -> Library:
        lib = self.get_library(lib_id)
        lib.name = name
        lib.metadata = metadata
        self.repo.update(lib)
        return lib

    def delete_library(self, lib_id: str) -> None:
        self.get_library(lib_id)
        self.repo.delete(lib_id)

    def create_document(
        self,
        lib_id: str,
        doc_id: UUID,
        title: str,
        metadata: Dict[str, Any]
    ) -> Document:
        lib = self.get_library(lib_id)
        if any(d.id == doc_id for d in lib.documents):
            raise ValueError("Document already exists")
        doc = Document(id=doc_id, title=title, chunks=[], metadata=metadata)
        lib.documents.append(doc)
        self.repo.update(lib)
        return doc

    def add_document(self, lib_id: str, doc_id: UUID, title: str, metadata: Dict[str, Any]) -> None:
        lib = self.get_library(lib_id)
        doc = Document(id=doc_id, title=title, chunks=[], metadata=metadata)
        lib.documents.append(doc)
        self.repo.update(lib)

    def list_documents(self, lib_id: str) -> List[Document]:
        return self.get_library(lib_id).documents

    def add_chunk(self, lib_id: str, doc_id: UUID, text: str, embedding: List[float], metadata: Dict[str, Any]) -> Chunk:
        lib = self.get_library(lib_id)
        doc = next((d for d in lib.documents if d.id == doc_id), None)
        if doc is None:
            raise ValueError('Document not found')
        chunk = Chunk(id=uuid4(), text=text,
                      embedding=embedding, metadata=metadata)
        doc.chunks.append(chunk)
        # print(f"Library after adding chunk: {lib}")
        self.repo.update(lib)
        print(f"Chunk: {chunk}")
        return chunk

    def list_chunks(self, lib_id: str, limit: int = 100, offset: int = 0) -> List[Chunk]:
        chunks = [c for d in self.get_library(
            lib_id).documents for c in d.chunks]
        return paginate(chunks, offset, limit)

    def update_chunk(
        self,
        lib_id: str,
        chunk_id: UUID,
        text: Optional[str],
        embedding: Optional[List[float]],
        metadata: Optional[Dict[str, Any]]
    ) -> Chunk:
        lib = self.get_library(lib_id)

        for doc in lib.documents:
            for chunk in doc.chunks:
                if chunk.id == chunk_id:
                    if text is not None:
                        chunk.text = text
                    if embedding is not None:
                        chunk.embedding = embedding
                    if metadata is not None:
                        chunk.metadata = metadata
                    # print(f"Updating chunk: {chunk}")
                    self.repo.update(lib)
                    return chunk
        raise ValueError("Chunk not found")

    def delete_chunk(self, lib_id: str, chunk_id: UUID) -> None:
        lib = self.get_library(lib_id)
        for d in lib.documents:
            for i, c in enumerate(d.chunks):
                if c.id == chunk_id:
                    del d.chunks[i]
                    self.repo.update(lib)
                    return
        raise ValueError('Chunk not found')

    def search(self, lib_id: str, query_embedding: List[float], k: int = 1, algorithm: str = 'kd', metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        lib = self.get_library(lib_id)
        chunks = [c for d in lib.documents for c in d.chunks]
        if metadata_filter:
            chunks = [c for c in chunks if all(c.metadata.get(
                k) == v for k, v in metadata_filter.items())]
        embeddings = [c.embedding for c in chunks]
        idx_class = {'kd': KDTree, 'ball': BallTree,
                     'linear': LinearIndex}[algorithm]
        index = idx_class(embeddings)
        idxs = index.nearest(query_embedding, k)
        results = []
        for idx in idxs:
            c = chunks[idx]
            dist = float(np.linalg.norm(
                np.array(query_embedding) - np.array(c.embedding)))
            results.append({"chunk": c, "distance": dist})
        return results
