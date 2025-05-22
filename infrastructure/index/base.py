from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

IndexType = np.float32


class BaseIndex(ABC):
    """Base class for all index implementations."""
    
    @abstractmethod
    def __init__(self, data: List[List[float]], **kwargs) -> None: ...
    
    @abstractmethod
    def nearest(
        self,
        target: Union[List[float], np.ndarray],
        k: int = 1
    ) -> List[IndexType]: ...