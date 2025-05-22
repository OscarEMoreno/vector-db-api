import os
import pickle
from threading import Lock
from typing import Dict, List, Optional

from domain.models import Library
from .base import BaseLibraryRepository


class PickleLibraryRepository(BaseLibraryRepository):
    def __init__(self, file_path: str = "data.pkl") -> None:
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

    def add(self, lib: Library) -> Library:
        with self._lock:
            self._data[str(lib.id)] = lib
            self._persist()
        return lib

    def get(self, lib_id: str) -> Optional[Library]:
        return self._data.get(lib_id)

    def update(self, lib: Library) -> Library:
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