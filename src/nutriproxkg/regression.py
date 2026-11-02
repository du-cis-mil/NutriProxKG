from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import numpy as np
from .eu_tolerance import eu_accuracy
from .regression_metrics import rmse_per_target


def _impute_train(Y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fill NaN in Y_train with per-column training means.

    Returns
    -------
    Y_train_imp : imputed matrix (no NaN)
    col_means   : training column means used for imputation
    """
    col_means = np.nanmean(Y_train, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_mask = ~np.isfinite(Y_train)
    Y_train_imp = Y_train.copy()
    Y_train_imp[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return Y_train_imp, col_means


def _per_nutrient_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: List[str],
    tol_cfg: Dict,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-nutrient euACC and RMSE, skipping NaN positions in y_true."""
    eu_per: Dict[str, float] = {}
    rmse_per_: Dict[str, float] = {}
    for idx, nutrient in enumerate(targets):
        valid = np.isfinite(y_true[:, idx])
        if valid.sum() == 0:
            eu_per[nutrient] = float("nan")
            rmse_per_[nutrient] = float("nan")
            continue
        y_t = y_true[valid, idx : idx + 1]
        y_p = y_pred[valid, idx : idx + 1]
        eu_scores = eu_accuracy(y_t, y_p, [nutrient], tol_cfg)
        rmse_scores = rmse_per_target(y_t, y_p, [nutrient])
        eu_per[nutrient] = float(eu_scores[f"eu_acc_{nutrient}"])
        rmse_per_[nutrient] = float(rmse_scores[f"rmse_{nutrient}"])
    return eu_per, rmse_per_
