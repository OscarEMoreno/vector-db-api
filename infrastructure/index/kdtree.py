import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import heapq
from .base import BaseIndex, IndexType


@dataclass
class KDNode:
    points: np.ndarray
    indices: np.ndarray
    axis: int
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None


class KDTree(BaseIndex):
    """
    Build: O(n log n)
    Query: average O(log n), worst-case O(n)
    """

    def __init__(
        self, 
        data: List[List[float]], 
        leaf_size: int = 40, 
        **kwargs
    ) -> None:
        self.data = np.array(data, dtype=np.float32)
        self.leaf_size = leaf_size
        self.dimensions = self.data.shape[1]
        self.root = self._build(np.arange(len(data)), depth=0)

    def _select_axis(self, points: np.ndarray, depth: int) -> int:
        if points.shape[0] < self.dimensions * 4:
            return depth % self.dimensions
        variances = np.var(points, axis=0)
        return np.argmax(variances)

    def _build(self, idxs: np.ndarray, depth: int) -> Optional[KDNode]:
        if len(idxs) == 0:
            return None
        points = self.data[idxs]
        
        if len(idxs) <= self.leaf_size:
            return KDNode(points=points, indices=idxs, axis=depth % self.dimensions)

        axis = self._select_axis(points, depth)
        median_idx = len(idxs) // 2
        
        partition_idx = np.argpartition(points[:, axis], median_idx)
        left_idxs = idxs[partition_idx[:median_idx]]
        right_idxs = idxs[partition_idx[median_idx + 1:]]
        median_idx_actual = partition_idx[median_idx]
        
        return KDNode(
            points=points[median_idx_actual:median_idx_actual+1],
            indices=idxs[median_idx_actual:median_idx_actual+1],
            axis=axis,
            left=self._build(left_idxs, depth + 1),
            right=self._build(right_idxs, depth + 1)
        )

    def nearest(
        self,
        target: Union[List[float], np.ndarray],
        k: int = 1
    ) -> List[IndexType]:
        target = np.array(target, dtype=np.float32)
        heap: List[Tuple[float, IndexType]] = []
        
        def squared_distance(p1: np.ndarray, p2: np.ndarray) -> float:
            diff = p1 - p2
            return np.dot(diff, diff)

        def search(node: Optional[KDNode]) -> None:
            if node is None:
                return
            for point, idx in zip(node.points, node.indices):
                dist = np.sqrt(squared_distance(target, point))
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, idx))
                elif -dist > heap[0][0]:
                    heapq.heapreplace(heap, (-dist, idx))
            
            if node.left is None and node.right is None:
                return
            axis_dist = target[node.axis] - node.points[0][node.axis]
            if axis_dist < 0:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
                
            search(first)
            if len(heap) < k or axis_dist * axis_dist < -heap[0][0]:
                search(second)
                
        search(self.root)
        return [idx for _, idx in sorted(heap, reverse=True)]
