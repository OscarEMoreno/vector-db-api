import os
import pytest
from uuid import uuid4
from fastapi.testclient import TestClient

from app.main import app
from domain.models import Library
from infrastructure.repositories import (
    JSONLibraryRepository,
    PickleLibraryRepository,
    SQLiteLibraryRepository,
    RepositoryFactory,
)
from infrastructure.leader_follower import LeaderFollowerRepository
from client.sdk import VectorDBClient

client = TestClient(app)


def create_library(api, name="TestLib", metadata=None):
    resp = api.post(
        "/libraries", json={"name": name, "metadata": metadata or {}})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    return data["id"]


# Core API CRUD & Search Tests (KD, Ball, Linear)
@pytest.mark.parametrize("algo", ["kd", "ball", "linear"])
def test_crud_and_search_algorithms(algo):
    # Create
    lib_id = create_library(client)

    # Read
    get_resp = client.get(f"/libraries/{lib_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == lib_id

    # Update
    upd = client.put(
        f"/libraries/{lib_id}", json={"name": "LibUpdated", "metadata": {"x": 1}}
    )
    assert upd.status_code == 200
    assert upd.json()["name"] == "LibUpdated"

    # Create Document
    doc_id = uuid4()
    doc_resp = client.post(
        f"/libraries/{lib_id}/documents",
        json={"id": str(doc_id), "title": "Doc1", "metadata": {"author": "A"}}
    )
    assert doc_resp.status_code == 200

    # Add Chunk
    embedding = [0.24475098, 0.33691406,
                 0.015457153, 0.12213135, -9.1552734e-05]
    ch = client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "doc_id": str(doc_id),
            "text": "hello",
            "embedding": embedding,
            "metadata": {"tag": "T"}
        }
    )
    assert ch.status_code == 200
    chunk = ch.json()
    assert "id" in chunk
    chunk_id = chunk["id"]

    # List Chunks with pagination
    chs = client.get(f"/libraries/{lib_id}/chunks")
    assert chs.status_code == 200
    assert isinstance(chs.json(), list)
    assert any(c["id"] == chunk_id for c in chs.json())

    # Update Chunk
    upc = client.put(
        f"/libraries/{lib_id}/chunks/{chunk_id}",
        json={
            "text": "updated text",

        }
    )
    assert upc.status_code == 200
    assert upc.json()["text"] == "updated text"

    # Search (self-match)
    sr = client.post(
        f"/libraries/{lib_id}/search",
        json={"embedding": embedding, "k": 1, "algorithm": algo}
    )
    assert sr.status_code == 200
    results = sr.json()["results"]
    assert len(results) == 1
    assert pytest.approx(0.0) == results[0]["distance"]

    # Invalid algorithm
    bad = client.post(
        f"/libraries/{lib_id}/search",
        json={"embedding": embedding, "k": 1, "algorithm": "unknown"}
    )
    assert bad.status_code == 422

    # Delete Chunk
    assert client.delete(
        f"/libraries/{lib_id}/chunks/{chunk_id}").status_code == 200

    # Delete Library
    assert client.delete(f"/libraries/{lib_id}").status_code == 200


# Metadata Filter Test
def test_search_with_metadata_filter(tmp_path, monkeypatch):
    # Isolate storage
    monkeypatch.chdir(tmp_path)
    local = TestClient(app)

    lib_id = create_library(local, "L", {})
    doc_id = uuid4()
    local.post(f"/libraries/{lib_id}/documents",
               json={"id": str(doc_id), "title": "D", "metadata": {}})
    emb = [0, 0, 0]
    keep = local.post(
        f"/libraries/{lib_id}/chunks",
        json={"doc_id": str(doc_id), "text": "A",
              "embedding": emb, "metadata": {"tag": "keep"}}
    ).json()["id"]
    local.post(
        f"/libraries/{lib_id}/chunks",
        json={"doc_id": str(doc_id), "text": "B",
              "embedding": emb, "metadata": {"tag": "drop"}}
    )
    sr = local.post(
        f"/libraries/{lib_id}/search",
        json={"embedding": emb, "k": 1, "algorithm": "linear",
              "metadata_filter": {"tag": "keep"}}
    )
    assert sr.status_code == 200
    assert sr.json()["results"][0]["chunk"]["id"] == keep


# Health Check
def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# Repository Unit Tests
@pytest.mark.parametrize("cls,path", [
    (JSONLibraryRepository, "data.json"),
    (PickleLibraryRepository, "data.pkl"),
    (SQLiteLibraryRepository, "data.db"),
])
def test_repo_crud(tmp_path, cls, path):
    os.chdir(tmp_path)
    repo = cls(path)
    lib = Library(id=uuid4(), name="R", documents=[], metadata={})
    repo.add(lib)
    assert repo.get(str(lib.id)).id == lib.id
    lib.name = "R2"
    repo.update(lib)
    assert repo.get(str(lib.id)).name == "R2"
    assert any(l.id == lib.id for l in repo.list_all())
    repo.delete(str(lib.id))
    assert repo.get(str(lib.id)) is None


# LibraryRepository Wrapper Test
def test_library_repository(tmp_path):
    os.chdir(tmp_path)
    for backend in ("json", "pickle", "sqlite"):
        repo = RepositoryFactory.create(
            backend_type=backend,
            json_path="data.json",
            pickle_path="data.pkl",
            sqlite_path="data.db"
        )
        lib = Library(id=uuid4(), name=backend, documents=[], metadata={})
        repo.add(lib)
        assert repo.get(str(lib.id)).id == lib.id
        repo.delete(str(lib.id))
        assert repo.get(str(lib.id)) is None


# Leader-Follower Replication Tests
def test_leader_follower(tmp_path):
    os.chdir(tmp_path)
    leader = JSONLibraryRepository("leader.json")
    f1 = JSONLibraryRepository("f1.json")
    f2 = JSONLibraryRepository("f2.json")
    lf = LeaderFollowerRepository(leader, [f1, f2])
    lib = Library(id=uuid4(), name="LF", documents=[], metadata={})
    lf.add(lib)
    for r in (leader, f1, f2):
        assert r.get(str(lib.id)).id == lib.id
    lib.name = "LF2"
    lf.update(lib)
    for r in (leader, f1, f2):
        assert r.get(str(lib.id)).name == "LF2"
    lf.delete(str(lib.id))
    for r in (leader, f1, f2):
        assert r.get(str(lib.id)) is None



# Python SDK Tests
# def test_sdk_flow():
#     sdk = VectorDBClient(base_url="http://localhost:8000")
#     lib = sdk.create_library("SDKLib", {"k": 1})
#     lib_id = lib['id']
#     assert sdk.get_library(lib_id)['name'] == "SDKLib"
#     doc_id = uuid4()
#     sdk.create_document(lib_id, doc_id, "Doc for SDK", {})
#     ch = sdk.add_chunk(lib_id, doc_id, "hello from SDK", [1, 2, 3], {})
#     res = sdk.search(lib_id, [1, 2, 3], k=1, algorithm='linear')
#     assert isinstance(res, list)
#     sdk.delete_chunk(lib_id, ch['id'])
#     sdk.delete_library(lib_id)
