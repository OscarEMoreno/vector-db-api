from uuid import UUID
from typing import List, Dict, Any
from pydantic import BaseModel


class Chunk(BaseModel):
    id: UUID
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

    class Config:
        from_attributes = True


class Document(BaseModel):
    id: UUID
    title: str
    chunks: List[Chunk]
    metadata: Dict[str, Any]

    class Config:
        from_attributes = True


class Library(BaseModel):
    id: UUID
    name: str
    documents: List[Document]
    metadata: Dict[str, Any]

    class Config:
        from_attributes = True
