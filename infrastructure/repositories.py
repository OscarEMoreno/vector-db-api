import os
import json
import pickle
import sqlite3
from threading import Lock
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from domain.models import Library


class BaseLibraryRepository(ABC):
    @abstractmethod
    def add(self, lib: Library) -> None: ...
    @abstractmethod
    def get(self, lib_id: str) -> Optional[Library]: ...
    @abstractmethod
    def update(self, lib: Library) -> None: ...
    @abstractmethod
    def delete(self, lib_id: str) -> None: ...
    @abstractmethod
    def list_all(self) -> List[Library]: ...


class JSONLibraryRepository(BaseLibraryRepository):
    def __init__(self, file_path: str = "data.json"):
        self.file_path = file_path
        self._lock = Lock()
        self._data: Dict[str, Library] = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                raw = json.load(f)
            for lib_dict in raw:
                lib = Library(**lib_dict)
                self._data[str(lib.id)] = lib

    def _persist(self) -> None:
        tmp = f"{self.file_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(
                [lib.model_dump() for lib in self._data.values()],
                f,
                indent=2,
                default=str,
            )
        os.replace(tmp, self.file_path)

    def add(self, lib: Library) -> None:
        with self._lock:
            self._data[str(lib.id)] = lib
            self._persist()
        return lib

    def get(self, lib_id: str) -> Optional[Library]:
        return self._data.get(lib_id)

    def update(self, lib: Library) -> None:
        with self._lock:
            self._data[str(lib.id)] = lib
            self._persist()
        return lib

    def delete(self, lib_id: str) -> None:
        with self._lock:
            self._data.pop(lib_id, None)
            self._persist()

    def list_all(self) -> List[Library]:
        return list(self._data.values())


class PickleLibraryRepository(BaseLibraryRepository):
    def __init__(self, file_path: str = "data.pkl"):
        self.file_path = file_path
        self._lock = Lock()
        self._data: Dict[str, Library] = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                self._data = pickle.load(f)

    def _persist(self) -> None:
        tmp = f"{self.file_path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._data, f)
        os.replace(tmp, self.file_path)

    def add(self, lib: Library) -> None:
        with self._lock:
            self._data[str(lib.id)] = lib
            self._persist()
        return lib

    def get(self, lib_id: str) -> Optional[Library]:
        return self._data.get(lib_id)

    def update(self, lib: Library) -> None:
        with self._lock:
            self._data[str(lib.id)] = lib
            self._persist()
        return lib

    def delete(self, lib_id: str) -> None:
        with self._lock:
            self._data.pop(lib_id, None)
            self._persist()

    def list_all(self) -> List[Library]:
        return list(self._data.values())


class SQLiteLibraryRepository(BaseLibraryRepository):
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._lock = Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS libraries (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _serialize(self, lib: Library) -> str:
        return json.dumps(lib.dict(), default=str)

    def _deserialize(self, txt: str) -> Library:
        return Library(**json.loads(txt))

    def add(self, lib: Library) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO libraries (id, data) VALUES (?, ?)",
                (str(lib.id), self._serialize(lib))
            )
            self._conn.commit()
        return lib

    def get(self, lib_id: str) -> Optional[Library]:
        cur = self._conn.execute(
            "SELECT data FROM libraries WHERE id = ?", (lib_id,)
        )
        row = cur.fetchone()
        return self._deserialize(row[0]) if row else None

    def update(self, lib: Library) -> None:
        self.add(lib)
        return lib

    def delete(self, lib_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM libraries WHERE id = ?", (lib_id,)
            )
            self._conn.commit()

    def list_all(self) -> List[Library]:
        cur = self._conn.execute("SELECT data FROM libraries")
        return [self._deserialize(row[0]) for row in cur.fetchall()]


class LibraryRepository(BaseLibraryRepository):
    def __init__(
        self,
        backend_type: str = "json",
        json_path: str = "data.json",
        pickle_path: str = "data.pkl",
        sqlite_path: str = "data.db"
    ):
        bt = backend_type.lower()
        if bt == "json":
            self._repo = JSONLibraryRepository(json_path)
        elif bt == "pickle":
            self._repo = PickleLibraryRepository(pickle_path)
        elif bt in ("sqlite", "db", "sql"):
            self._repo = SQLiteLibraryRepository(sqlite_path)
        else:
            raise ValueError(f"Unknown backend_type '{backend_type}'")

    def add(self, lib: Library) -> None:
        self._repo.add(lib)
        return lib

    def get(self, lib_id: str) -> Optional[Library]:
        return self._repo.get(lib_id)

    def update(self, lib: Library) -> None:
        self._repo.update(lib)
        return lib

    def delete(self, lib_id: str) -> None:
        self._repo.delete(lib_id)

    def list_all(self) -> List[Library]:
        return self._repo.list_all()
