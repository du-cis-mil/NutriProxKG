from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import mean_squared_error


def rmse_per_target(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: Sequence[str]
) -> Dict[str, float]:
    """RMSE per target and mean RMSE."""
    out: Dict[str, float] = {}
    per_target = []
    for idx, name in enumerate(target_names):
        rmse = float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx])))
        out[f"rmse_{name}"] = rmse
        per_target.append(rmse)
    out["rmse_mean"] = float(np.mean(per_target)) if per_target else float("nan")
    return out
