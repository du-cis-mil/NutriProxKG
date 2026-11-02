"""Cross-validation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def safe_stratified_splits(y: np.ndarray, requested_k: int) -> int:
    """Compute maximum valid k for stratified folds."""
    if y.size == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        return 0
    return int(max(0, min(int(requested_k), int(y.shape[0]), int(counts.min()))))


def make_stratified_kfold(
    y: np.ndarray, requested_k: int, seed: int
) -> Optional[StratifiedKFold]:
    """Build stratified k-fold splitter safely."""
    k = safe_stratified_splits(y, requested_k)
    if k < 2:
        return None
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def make_kfold(n_samples: int, requested_k: int, seed: int) -> Optional[KFold]:
    """Build standard k-fold splitter safely."""
    k = int(min(max(0, requested_k), max(0, n_samples)))
    if k < 2:
        return None
    return KFold(n_splits=k, shuffle=True, random_state=seed)
