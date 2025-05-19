import numpy as np


class LinearIndex:
    """
    Build: O(1)
    Query: O(n)
    """

    def __init__(self, data):
        self.data = data

    def nearest(self, target, k=1):
        dists = [
            (float(np.linalg.norm(np.array(target) - np.array(pt))), idx)
            for idx, pt in enumerate(self.data)
        ]
        dists.sort(key=lambda x: x[0])
        return [idx for _, idx in dists[:k]]
