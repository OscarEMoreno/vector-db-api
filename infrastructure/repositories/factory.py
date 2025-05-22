from typing import Type, Dict

from .base import BaseLibraryRepository
from .json_repo import JSONLibraryRepository
from .pickle_repo import PickleLibraryRepository
from .sqlite_repo import SQLiteLibraryRepository


class RepositoryFactory:    
    _repo_types: Dict[str, Type[BaseLibraryRepository]] = {
        'json': JSONLibraryRepository,
        'pickle': PickleLibraryRepository,
        'sqlite': SQLiteLibraryRepository,
        'sql': SQLiteLibraryRepository,
        'db': SQLiteLibraryRepository,
    }
    @classmethod
    def create(
        cls,
        backend_type: str = "json",
        json_path: str = "data.json",
        pickle_path: str = "data.pkl",
        sqlite_path: str = "data.db"
    ) -> BaseLibraryRepository:
        bt = backend_type.lower()
        if bt not in cls._repo_types:
            raise ValueError(
                f"Unsupported backend_type '{backend_type}'. "
                f"Supported types are: {list(cls._repo_types.keys())}"
            )
        
        repo_class = cls._repo_types[bt]
        if bt in ('sqlite', 'sql', 'db'):
            return repo_class(sqlite_path)
        elif bt == 'pickle':
            return repo_class(pickle_path)
        else:  # json
            return repo_class(json_path) 