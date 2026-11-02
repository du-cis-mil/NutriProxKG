import numpy as np
from sklearn.preprocessing import normalize


def cosine_edges(X, eligible, theta, batch=256):
    ids = np.flatnonzero(eligible)
    if ids.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    unit = normalize(X[ids], norm="l2")
    left = []
    right = []
    for start in range(0, len(ids), batch):
        scores = (unit[start : start + batch] @ unit.T).tocoo()
        hit = (scores.data >= theta) & (scores.col > scores.row + start)
        left.extend(ids[scores.row[hit] + start].tolist())
        right.extend(ids[scores.col[hit]].tolist())
    return np.asarray([left, right], dtype=np.int64)
