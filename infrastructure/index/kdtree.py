import numpy as np


class KDNode:
    def __init__(self, point, idx, left=None, right=None):
        self.point = point
        self.idx = idx
        self.left = left
        self.right = right


class KDTree:
    """
    Build: O(n log n)
    Query: average O(log n), worst-case O(n)
    """

    def __init__(self, data):
        def build(points, depth=0):
            if not points:
                return None
            k = len(points[0][0])
            axis = depth % k
            points.sort(key=lambda x: x[0][axis])
            mid = len(points) // 2
            node = KDNode(
                points[mid][0],
                points[mid][1],
                build(points[:mid], depth+1),
                build(points[mid+1:], depth+1)
            )
            return node

        pts = [(d, i) for i, d in enumerate(data)]
        self.root = build(pts)

    def nearest(self, target, k=1):
        best = []

        def search(node, depth=0):
            if node is None:
                return
            dist = np.linalg.norm(np.array(target) - np.array(node.point))
            best.append((dist, node.idx))
            best.sort(key=lambda x: x[0])
            if len(best) > k:
                best.pop()  # keep only k closest
            axis = depth % len(target)
            diff = target[axis] - node.point[axis]
            close, away = (node.left, node.right) if diff < 0 else (
                node.right, node.left)
            search(close, depth+1)
            if len(best) < k or abs(diff) < best[-1][0]:
                search(away, depth+1)

        search(self.root)
        return [idx for _, idx in best]
