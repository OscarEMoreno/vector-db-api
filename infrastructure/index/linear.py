import numpy as np
from typing import List, Union
from .base import BaseIndex, IndexType


class LinearIndex(BaseIndex):
    """
    Build: O(1)
    Query: O(n)
    """
    def __init__(self, data: List[List[float]], **kwargs) -> None:
        self.data = np.array(data, dtype=np.float32)

    def nearest(
        self, 
        target: Union[List[float], np.ndarray], 
        k: int = 1
    ) -> List[IndexType]:

        target = np.array(target, dtype=np.float32)
        dists = [
            (float(np.linalg.norm(target - pt)), i)
            for i, pt in enumerate(self.data)
        ]
        dists.sort(key=lambda x: x[0])
        return [i for _, i in dists[:k]]
