import json
import sqlite3
from threading import Lock
from typing import List, Optional

from domain.models import Library
from .base import BaseLibraryRepository


class SQLiteLibraryRepository(BaseLibraryRepository):
    def __init__(self, db_path: str = "data.db") -> None:
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
        return json.dumps(lib.model_dump(), default=str)

    def _deserialize(self, txt: str) -> Library:
        return Library(**json.loads(txt))

    def add(self, lib: Library) -> Library:
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

    def update(self, lib: Library) -> Library:
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