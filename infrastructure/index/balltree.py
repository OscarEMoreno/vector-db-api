import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import heapq
from .base import BaseIndex, IndexType


@dataclass
class BallNode:
    points_idx: np.ndarray
    center: np.ndarray
    radius: float
    left: Optional['BallNode'] = None
    right: Optional['BallNode'] = None


class BallTree(BaseIndex):
    """
    Build: O(n log n)
    Query: average O(log n)
    """

    def __init__(
        self, 
        data: List[List[float]], 
        leaf_size: int = 40, 
        **kwargs
    ) -> None:
        self.data = np.array(data, dtype=np.float32)
        self.leaf_size = leaf_size
        self.root = self._build(np.arange(len(data)))

    def _build(
        self, 
        idxs: np.ndarray
    ) -> Optional[BallNode]:
        if len(idxs) == 0:
            return None
            
        if len(idxs) <= self.leaf_size:
            points = self.data[idxs]
            center = np.mean(points, axis=0)
            radius = np.max(np.linalg.norm(points - center, axis=1))
            return BallNode(idxs, center, radius)

        points = self.data[idxs]
        var = np.var(points, axis=0)
        split_dim = np.argmax(var)
        median_idx = len(idxs) // 2
        partition_idx = np.argpartition(points[:, split_dim], median_idx)
        left_idxs = idxs[partition_idx[:median_idx]]
        right_idxs = idxs[partition_idx[median_idx:]]
        center = np.mean(points, axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        
        return BallNode(
            points_idx=idxs,
            center=center,
            radius=radius,
            left=self._build(left_idxs),
            right=self._build(right_idxs)
        )

    def nearest(
        self, 
        target: Union[List[float], np.ndarray], 
        k: int = 1
    ) -> List[IndexType]:
        target = np.array(target, dtype=np.float32)
        heap: List[Tuple[float, IndexType]] = []
        
        def search(node: Optional[BallNode]) -> None:
            if node is None:
                return
                
            dist_to_center = np.linalg.norm(target - node.center)
            if len(heap) == k and dist_to_center - node.radius > -heap[0][0]:
                return
            
            points = self.data[node.points_idx]
            dists = np.linalg.norm(points - target, axis=1)
            for dist, idx in zip(dists, node.points_idx):
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, idx))
                elif -dist > heap[0][0]:
                    heapq.heapreplace(heap, (-dist, idx))
            if node.left and node.right:
                left_dist = np.linalg.norm(target - node.left.center)
                right_dist = np.linalg.norm(target - node.right.center)
                if left_dist < right_dist:
                    search(node.left)
                    search(node.right)
                else:
                    search(node.right)
                    search(node.left)
            else:
                search(node.left)
                search(node.right)
        search(self.root)
        return [idx for _, idx in sorted(heap, reverse=True)]
