from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import numpy as np
from sklearn.preprocessing import StandardScaler
from .cv import make_kfold
from .eu_tolerance import load_tolerance_config
from .regression import _impute_train, _per_nutrient_metrics

FOCUS_NUTRIENTS = ("carbs_g", "fat_g", "protein_g")
FitPredict = Callable


def _aggregate(
    fold_eu: List[Dict[str, float]],
    fold_rmse: List[Dict[str, float]],
    targets: List[str],
    n: int,
    k: int,
) -> Dict:
    per_nutrient: Dict[str, Dict[str, float]] = {}
    for nut in targets:
        eu_vals = [f[nut] for f in fold_eu if np.isfinite(f.get(nut, float("nan")))]
        rmse_vals = [f[nut] for f in fold_rmse if np.isfinite(f.get(nut, float("nan")))]
        per_nutrient[nut] = {
            "euACC_mean": float(np.mean(eu_vals)) if eu_vals else float("nan"),
            "euACC_std": float(np.std(eu_vals)) if eu_vals else float("nan"),
            "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
            "rmse_std": float(np.std(rmse_vals)) if rmse_vals else float("nan"),
            "n_folds": max(len(eu_vals), len(rmse_vals)),
        }
    eu_all = [
        v["euACC_mean"] for v in per_nutrient.values() if np.isfinite(v["euACC_mean"])
    ]
    rmse_all = [
        v["rmse_mean"] for v in per_nutrient.values() if np.isfinite(v["rmse_mean"])
    ]
    block = {
        "status": "ok",
        "n_samples": int(n),
        "n_folds": int(k),
        "per_nutrient": per_nutrient,
        "euACC_mean_over_nutrients": float(np.mean(eu_all)) if eu_all else float("nan"),
        "rmse_mean_over_nutrients": float(np.mean(rmse_all))
        if rmse_all
        else float("nan"),
    }
    return _enrich(block)


def aggregate_seed_blocks(
    blocks: List[Dict], seeds: List[int], targets: List[str]
) -> Dict:
    """Average per-seed model blocks → mean ± std-over-seeds (reproducibility).

    Each block in ``blocks`` is a kfold_regression output for one seed (the fold
    split for a given seed is identical across all graphs/models, so the seed
    variance reflects fold + GPU-nondeterminism, not food-set differences).
    """
    ok = [(s, b) for s, b in zip(seeds, blocks) if b.get("status") == "ok"]
    if not ok:
        return {"status": "no_seeds"}

    def ms(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return (
            (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))
        )

    eu_m, eu_s = ms([b["euACC_mean_over_nutrients"] for _, b in ok])
    rm_m, rm_s = ms([b["rmse_mean_over_nutrients"] for _, b in ok])

    per: Dict[str, Dict[str, float]] = {}
    for nut in targets:
        em, es = ms(
            [b.get("per_nutrient", {}).get(nut, {}).get("euACC_mean") for _, b in ok]
        )
        rm2, rs = ms(
            [b.get("per_nutrient", {}).get(nut, {}).get("rmse_mean") for _, b in ok]
        )
        per[nut] = {
            "euACC_mean": em,
            "euACC_std": es,
            "euACC_seed_std": es,
            "rmse_mean": rm2,
            "rmse_std": rs,
            "rmse_seed_std": rs,
        }

    block = {
        "status": "ok",
        "n_seeds": len(ok),
        "seeds": [s for s, _ in ok],
        "euACC_mean_over_nutrients": eu_m,
        "euACC_seed_std": eu_s,
        "rmse_mean_over_nutrients": rm_m,
        "rmse_seed_std": rm_s,
        "per_nutrient": per,
        "per_seed": [
            {
                "seed": s,
                "euACC": b["euACC_mean_over_nutrients"],
                "rmse": b["rmse_mean_over_nutrients"],
            }
            for s, b in ok
        ],
    }
    return _enrich(block)


def _enrich(block: Dict) -> Dict:
    """Add focus_nutrients + other_nutrients_mean (matches runner._enrich_task2)."""
    pn = block.get("per_nutrient", {})
    focus, other_eu, other_rmse = {}, [], []
    for nut, st in pn.items():
        if nut in FOCUS_NUTRIENTS:
            focus[nut] = {
                "euACC_mean": st.get("euACC_mean"),
                "euACC_std": st.get("euACC_std"),
                "rmse_mean": st.get("rmse_mean"),
                "rmse_std": st.get("rmse_std"),
            }
        else:
            if np.isfinite(st.get("euACC_mean", float("nan"))):
                other_eu.append(st["euACC_mean"])
            if np.isfinite(st.get("rmse_mean", float("nan"))):
                other_rmse.append(st["rmse_mean"])
    block["focus_nutrients"] = focus
    block["other_nutrients_mean"] = {
        "euACC_mean": float(np.mean(other_eu)) if other_eu else None,
        "rmse_mean": float(np.mean(other_rmse)) if other_rmse else None,
        "n_nutrients": len(other_eu),
    }
    return block


def kfold_regression(
    fit_predict: FitPredict,
    Y_raw: np.ndarray,
    targets: List[str],
    k: int = 5,
    seed: int = 42,
    tolerance_yaml: str | None = None,
    collect_preds: bool = False,
) -> Dict:
    """Run transductive k-fold regression and return a task2-style model block.

    If ``collect_preds``, attaches ``_pred_all`` (n, n_targets) holding each food's
    test-fold prediction in original units (food predicted exactly once per run).
    """
    n = Y_raw.shape[0]
    if tolerance_yaml is None:
        raise ValueError("An explicit tolerance YAML path is required")
    tol_cfg = load_tolerance_config(tolerance_yaml)
    kf = make_kfold(n_samples=n, requested_k=k, seed=seed)
    if kf is None:
        return {"status": "insufficient_data", "n_samples": int(n)}

    fold_eu, fold_rmse = [], []
    pred_all = (
        np.full((n, len(targets)), np.nan, dtype=np.float64) if collect_preds else None
    )
    for fold_idx, (train_rows, test_rows) in enumerate(kf.split(np.arange(n)), start=1):
        Y_train = Y_raw[train_rows]
        Y_train_imp, _ = _impute_train(Y_train)
        y_scaler = StandardScaler().fit(Y_train_imp)
        Y_train_std = y_scaler.transform(Y_train_imp).astype(np.float32)

        y_pred_std = np.asarray(
            fit_predict(fold_idx, train_rows, test_rows, Y_train_std), dtype=np.float64
        )
        y_pred = y_scaler.inverse_transform(y_pred_std).astype(np.float32)
        if collect_preds:
            pred_all[test_rows] = y_pred

        eu_per, rmse_per = _per_nutrient_metrics(
            Y_raw[test_rows], y_pred, targets, tol_cfg
        )
        fold_eu.append(eu_per)
        fold_rmse.append(rmse_per)

    block = _aggregate(fold_eu, fold_rmse, targets, n, k)
    if collect_preds:
        block["_pred_all"] = pred_all
    return block
