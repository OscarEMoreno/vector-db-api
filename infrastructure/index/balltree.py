import numpy as np


class BallNode:
    def __init__(self, points_idx, center, radius, left=None, right=None):
        self.points_idx = points_idx
        self.center = center
        self.radius = radius
        self.left = left
        self.right = right


class BallTree:
    """
    Build: O(n log n)
    Query: average O(log n)
    """

    def __init__(self, data):

        data = np.array(data, dtype=float)
        self.data = data

        def build(idxs):
            pts = [data[i] for i in idxs]
            center = np.mean(pts, axis=0)
            radius = max(np.linalg.norm(p - center) for p in pts)
            if len(idxs) <= 1:
                return BallNode(idxs, center, radius)
            # pick two farthest points to split
            a, b = max(
                ((i, j) for i in idxs for j in idxs if i < j),
                key=lambda pair: (
                    np.linalg.norm(data[pair[0]] - center)
                    + np.linalg.norm(data[pair[1]] - center)
                )
            )
            left_idxs = [
                i for i in idxs
                if np.linalg.norm(data[i] - data[a]) < np.linalg.norm(data[i] - data[b])
            ]
            right_idxs = [i for i in idxs if i not in left_idxs]
            return BallNode(idxs, center, radius, build(left_idxs), build(right_idxs))

        self.root = build(list(range(len(data))))

    def nearest(self, target, k=1):
        target = np.array(target, dtype=float)
        data = self.data
        best: list[tuple[float, int]] = []

        def search(node):
            if node is None:
                return
            dist_center = np.linalg.norm(target - node.center)

            if len(best) == k and dist_center - node.radius > best[-1][0]:
                return

            for idx in node.points_idx:
                d = np.linalg.norm(target - data[idx])
                best.append((d, idx))

            # keep only the top‐k
            best.sort(key=lambda x: x[0])
            if len(best) > k:
                best[:] = best[:k]

            search(node.left)
            search(node.right)

        search(self.root)
        return [idx for _, idx in best]
