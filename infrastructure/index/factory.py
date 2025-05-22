
from typing import Type, Dict, List
from .kdtree import KDTree
from .balltree import BallTree
from .linear import LinearIndex
from .base import BaseIndex

class IndexFactory:    
    _index_types: Dict[str, Type[BaseIndex]] = {
        'kd': KDTree,
        'ball': BallTree,
        'linear': LinearIndex
    }

    @classmethod
    def create(cls, algorithm: str, data: List[List[float]], **kwargs) -> BaseIndex:
        if algorithm not in cls._index_types:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Supported types are: {list(cls._index_types.keys())}"
            )
        
        index_class = cls._index_types[algorithm]
        return index_class(data, **kwargs) 