import os
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from typing import List
from uuid import UUID

from app.schemas import (
    LibraryCreate,
    DocumentCreate,
    ChunkCreate,
    ChunkUpdate,
    SearchRequest,
)
from app.services import LibraryService
from infrastructure.repositories import BaseLibraryRepository, RepositoryFactory
from domain.models import Document, Library, Chunk


def get_repository() -> BaseLibraryRepository:
    return RepositoryFactory.create(
        backend_type=os.getenv('REPO_TYPE', 'json'),
        json_path=os.getenv('JSON_PATH', 'data.json'),
        pickle_path=os.getenv('PICKLE_PATH', 'data.pkl'),
        sqlite_path=os.getenv('SQLITE_PATH', 'data.db')
    )


def get_service(repo: BaseLibraryRepository = Depends(get_repository)) -> LibraryService:
    return LibraryService(repo)


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/libraries", response_model=Library)
async def create_library(
    req: LibraryCreate,
    service: LibraryService = Depends(get_service)
) -> Library:
    return service.create_library(req.name, req.metadata)


@app.get("/libraries/{lib_id}")
async def read_library(
    lib_id: str,
    service: LibraryService = Depends(get_service)
) -> Library:
    try:
        return service.get_library(lib_id)
    except ValueError:
        raise HTTPException(404, "Library not found")


@app.put("/libraries/{lib_id}")
async def update_library(
    lib_id: str,
    req: LibraryCreate,
    service: LibraryService = Depends(get_service)
) -> Library:
    try:
        return service.update_library(lib_id, req.name, req.metadata)
    except ValueError:
        raise HTTPException(404, "Library not found")


@app.delete("/libraries/{lib_id}")
async def delete_library(
    lib_id: str,
    service: LibraryService = Depends(get_service)
) -> None:
    try:
        service.delete_library(lib_id)
    except ValueError:
        raise HTTPException(404, "Library not found")


@app.post("/libraries/{lib_id}/documents", response_model=Document)
async def create_document(
    lib_id: str,
    req: DocumentCreate,
    service: LibraryService = Depends(get_service)
) -> Document:
    try:
        return service.create_document(lib_id, req.id, req.title, req.metadata)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/libraries/{lib_id}/documents", response_model=List[DocumentCreate])
async def list_documents(
    lib_id: str,
    service: LibraryService = Depends(get_service)
) -> List[DocumentCreate]:
    return service.list_documents(lib_id)


@app.post("/libraries/{lib_id}/chunks", response_model=Chunk)
async def add_chunk(
    lib_id: str,
    req: ChunkCreate,
    service: LibraryService = Depends(get_service)
) -> Chunk:
    try:
        return service.add_chunk(
            lib_id,
            req.doc_id,
            req.text,
            req.embedding,
            req.metadata
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/libraries/{lib_id}/chunks", response_model=List[Chunk])  
async def list_chunks(
    lib_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, gt=0),
    service: LibraryService = Depends(get_service)
) -> List[Chunk]:
    return service.list_chunks(lib_id, limit, offset)


@app.put("/libraries/{lib_id}/chunks/{chunk_id}", response_model=ChunkUpdate)
async def update_chunk(
    lib_id: str,
    chunk_id: UUID,
    req: ChunkUpdate,
    service: LibraryService = Depends(get_service)
) -> ChunkUpdate:
    try:
        return service.update_chunk(
            lib_id,
            chunk_id,
            req.text,
            req.embedding,
            req.metadata
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.delete("/libraries/{lib_id}/chunks/{chunk_id}")
async def delete_chunk(
    lib_id: str,
    chunk_id: UUID,
    service: LibraryService = Depends(get_service)
) -> None:
    try:
        return service.delete_chunk(lib_id, chunk_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/libraries/{lib_id}/search")
async def search(
    lib_id: str,
    req: SearchRequest,
    service: LibraryService = Depends(get_service)
) -> dict:
    try:
        return {
            "results": service.search(
                lib_id,
                req.embedding,
                req.k,
                req.algorithm,
                req.metadata_filter
            )
        }
    except ValueError:
        raise HTTPException(404, "Library not found")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}
