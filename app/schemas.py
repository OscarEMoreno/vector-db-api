from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, field_validator, model_validator


class LibraryCreate(BaseModel):
    name: str
    metadata: Dict[str, Any]

    model_config = {"from_attributes": True}



class DocumentCreate(BaseModel):
    id: UUID
    title: str
    metadata: Dict[str, Any]

    model_config = {
        "from_attributes": True
    }


class ChunkCreate(BaseModel):
    doc_id: UUID
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

    # @field_validator("embedding")
    # def check_embedding_length(cls, v):
    #     if len(v) not in (64, 128, 256):
    #         raise ValueError("Embedding must be length 64, 128, or 256")
    #     return v

    model_config = {
        "from_attributes": True
    }


class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def at_least_one_field(cls, m: "ChunkUpdate") -> "ChunkUpdate":
        print(m)
        if m.text is None and m.embedding is None and m.metadata is None:
            raise ValueError("At least one field must be provided")
        return m

    model_config = {
        "from_attributes": True
    }


class SearchRequest(BaseModel):
    embedding: List[float]
    k: int = 1
    algorithm: str = "kd"
    metadata_filter: Optional[Dict[str, Any]] = None

    @field_validator("algorithm")
    def valid_algorithm(cls, v):
        if v not in ("kd", "ball", "linear"):
            raise ValueError("algorithm must be one of: kd, ball, linear")
        return v

    model_config = {
        "from_attributes": True
    }
