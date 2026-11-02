"""EU nutrition-label tolerance helpers for Task2.

Task2 uses the food-label tolerances from Section 3 / Table 1 of the EU
guidance document for foods other than food supplements.

Metric semantics
----------------
- ``y_pred`` is treated as the declared label value.
- ``y_true`` is treated as the control / measured value.
- A prediction is counted as correct when the measured value falls inside the
  EU tolerance band around the declared value.

The module still accepts the legacy piecewise YAML format used previously so
older configs do not break, but the default configuration now follows the EU
Table 1 rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import yaml

RULE_CARB_LIKE = "eu_table1_carb_like"
RULE_FAT = "eu_table1_fat"
RULE_FATTY_ACID = "eu_table1_fatty_acid"
RULE_SODIUM = "eu_table1_sodium"
RULE_SALT = "eu_table1_salt"
RULE_VITAMIN = "eu_table1_vitamin"
RULE_MINERAL = "eu_table1_mineral"
RULE_UNSUPPORTED = "unsupported"
RULE_LEGACY_PIECEWISE = "legacy_piecewise"

LEGACY_KEYS = {
    "low_threshold",
    "mid_threshold",
    "low_abs",
    "mid_ratio",
    "high_abs",
    "high_ratio",
}

MINERAL_NUTRIENTS = {
    "calcium_g",
    "chloride_g",
    "copper_g",
    "iodine_g",
    "iron_g",
    "magnesium_g",
    "manganese_g",
    "phosphorus_g",
    "potassium_g",
    "selenium_g",
    "zinc_g",
}


def _rule_entry(rule: str, note: str | None = None) -> Dict[str, object]:
    out: Dict[str, object] = {"rule": rule}
    if note:
        out["note"] = note
    return out


DEFAULT_TOLERANCE_CONFIG: Dict[str, Dict[str, object]] = {
    "carbs_g": _rule_entry(RULE_CARB_LIKE),
    "fat_g": _rule_entry(RULE_FAT),
    "protein_g": _rule_entry(RULE_CARB_LIKE),
    "water_g": _rule_entry(
        RULE_UNSUPPORTED,
        note="EU guidance Table 1 does not define a nutrition-label tolerance for water.",
    ),
    "calcium_g": _rule_entry(RULE_MINERAL),
    "copper_g": _rule_entry(RULE_MINERAL),
    "iodine_g": _rule_entry(RULE_MINERAL),
    "iron_g": _rule_entry(RULE_MINERAL),
    "manganese_g": _rule_entry(RULE_MINERAL),
    "phosphorus_g": _rule_entry(RULE_MINERAL),
    "potassium_g": _rule_entry(RULE_MINERAL),
    "selenium_g": _rule_entry(RULE_MINERAL),
    "sodium_g": _rule_entry(RULE_SODIUM),
    "vitamin_a_retinol_g": _rule_entry(RULE_VITAMIN),
    "vitamin_b1_thiamine_g": _rule_entry(RULE_VITAMIN),
    "vitamin_b2_riboflavin_g": _rule_entry(RULE_VITAMIN),
    "vitamin_b9_folate_g": _rule_entry(RULE_VITAMIN),
    "vitamin_c_g": _rule_entry(RULE_VITAMIN),
    "vitamin_e_tocopherol_g": _rule_entry(RULE_VITAMIN),
    "zinc_g": _rule_entry(RULE_MINERAL),
}


def _default_rule_for_nutrient(nutrient: str) -> Dict[str, object] | None:
    name = str(nutrient)
    if name in DEFAULT_TOLERANCE_CONFIG:
        return dict(DEFAULT_TOLERANCE_CONFIG[name])
    if name.startswith("vitamin_"):
        return _rule_entry(RULE_VITAMIN)
    if name in MINERAL_NUTRIENTS:
        return _rule_entry(RULE_MINERAL)
    return None


def _normalise_rule(
    nutrient: str,
    item: Mapping[str, object],
    default: Mapping[str, object] | None,
) -> Dict[str, object]:
    if LEGACY_KEYS & set(item.keys()):
        base = dict(default or {})
        out: Dict[str, object] = {"rule": RULE_LEGACY_PIECEWISE}
        for key in sorted(LEGACY_KEYS):
            raw = item.get(key, base.get(key))
            if raw is None:
                raise ValueError(
                    f"Missing legacy tolerance field '{key}' for nutrient '{nutrient}'"
                )
            out[key] = float(raw)
        if "note" in item or "note" in base:
            out["note"] = str(item.get("note", base.get("note")))
        return out

    rule = str(item.get("rule", (default or {}).get("rule", RULE_UNSUPPORTED)))
    out = {"rule": rule}

    if rule == RULE_UNSUPPORTED:
        note = item.get("note", (default or {}).get("note"))
        if note is not None:
            out["note"] = str(note)
        return out

    if rule == RULE_VITAMIN:
        out["lower_ratio"] = float(item.get("lower_ratio", 0.35))
        out["upper_ratio"] = float(item.get("upper_ratio", 0.50))
        return out

    if rule == RULE_MINERAL:
        out["lower_ratio"] = float(item.get("lower_ratio", 0.35))
        out["upper_ratio"] = float(item.get("upper_ratio", 0.45))
        return out

    if rule in {
        RULE_CARB_LIKE,
        RULE_FAT,
        RULE_FATTY_ACID,
        RULE_SODIUM,
        RULE_SALT,
    }:
        return out

    raise ValueError(f"Unsupported tolerance rule '{rule}' for nutrient '{nutrient}'")


def load_tolerance_config(path: str | Path) -> Dict[str, Dict[str, object]]:
    """Load YAML tolerance config, writing EU Table 1 defaults when missing."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            yaml.safe_dump(DEFAULT_TOLERANCE_CONFIG, sort_keys=True), encoding="utf-8"
        )
        return {k: dict(v) for k, v in DEFAULT_TOLERANCE_CONFIG.items()}

    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Tolerance config must be a mapping: {cfg_path}")

    merged: Dict[str, Dict[str, object]] = {}
    all_nutrients = set(DEFAULT_TOLERANCE_CONFIG.keys()) | {
        str(k) for k in payload.keys()
    }
    for nutrient in sorted(all_nutrients):
        default = _default_rule_for_nutrient(nutrient)
        item = payload.get(nutrient)
        if item is None:
            if default is not None:
                merged[nutrient] = default
            continue
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Tolerance config for nutrient '{nutrient}' must be a mapping"
            )
        merged[nutrient] = _normalise_rule(nutrient, item, default)
    return merged


def _clip_declared(values: np.ndarray) -> np.ndarray:
    declared = np.asarray(values, dtype=np.float32)
    return np.maximum(declared, 0.0).astype(np.float32)


def _legacy_tolerance_from_measured(
    y_true: np.ndarray, params: Mapping[str, object]
) -> np.ndarray:
    """Legacy symmetric tolerance around the measured value."""
    y = np.asarray(y_true, dtype=np.float32)
    low_threshold = float(params["low_threshold"])
    mid_threshold = float(params["mid_threshold"])
    low_abs = float(params["low_abs"])
    mid_ratio = float(params["mid_ratio"])
    high_abs = float(params["high_abs"])
    high_ratio = float(params["high_ratio"])

    tol = np.where(
        y <= low_threshold,
        low_abs,
        np.where(
            y <= mid_threshold,
            np.maximum(mid_ratio * y, low_abs),
            np.maximum(high_ratio * y, high_abs),
        ),
    )
    return tol.astype(np.float32)


def _ratio_bounds(
    declared: np.ndarray,
    lower_ratio: float,
    upper_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(0.0, declared * (1.0 - float(lower_ratio)))
    upper = declared * (1.0 + float(upper_ratio))
    return lower.astype(np.float32), upper.astype(np.float32)


def _declared_bounds(
    declared: np.ndarray,
    params: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray] | None:
    rule = str(params.get("rule", RULE_UNSUPPORTED))
    d = _clip_declared(declared)

    if rule == RULE_UNSUPPORTED:
        return None

    if rule == RULE_CARB_LIKE:
        tol = np.where(d < 10.0, 2.0, np.where(d <= 40.0, 0.20 * d, 8.0))
        return np.maximum(0.0, d - tol).astype(np.float32), (d + tol).astype(np.float32)

    if rule == RULE_FAT:
        tol = np.where(d < 10.0, 1.5, np.where(d <= 40.0, 0.20 * d, 8.0))
        return np.maximum(0.0, d - tol).astype(np.float32), (d + tol).astype(np.float32)

    if rule == RULE_FATTY_ACID:
        tol = np.where(d < 4.0, 0.8, 0.20 * d)
        return np.maximum(0.0, d - tol).astype(np.float32), (d + tol).astype(np.float32)

    if rule == RULE_SODIUM:
        tol = np.where(d < 0.5, 0.15, 0.20 * d)
        return np.maximum(0.0, d - tol).astype(np.float32), (d + tol).astype(np.float32)

    if rule == RULE_SALT:
        tol = np.where(d < 1.25, 0.375, 0.20 * d)
        return np.maximum(0.0, d - tol).astype(np.float32), (d + tol).astype(np.float32)

    if rule == RULE_VITAMIN:
        return _ratio_bounds(
            declared=d,
            lower_ratio=float(params.get("lower_ratio", 0.35)),
            upper_ratio=float(params.get("upper_ratio", 0.50)),
        )

    if rule == RULE_MINERAL:
        return _ratio_bounds(
            declared=d,
            lower_ratio=float(params.get("lower_ratio", 0.35)),
            upper_ratio=float(params.get("upper_ratio", 0.45)),
        )

    raise ValueError(f"Unsupported tolerance rule '{rule}'")


def compliance_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nutrient_names: Sequence[str],
    config: Mapping[str, Mapping[str, object]],
) -> np.ndarray:
    """Return compliance mask per target, using NaN for unsupported nutrients."""
    y_t = np.asarray(y_true, dtype=np.float32)
    y_p = np.asarray(y_pred, dtype=np.float32)
    if y_t.shape != y_p.shape:
        raise ValueError(f"Shape mismatch: y_true={y_t.shape}, y_pred={y_p.shape}")

    cols = []
    for idx, nutrient in enumerate(nutrient_names):
        params = (
            config.get(nutrient)
            or _default_rule_for_nutrient(str(nutrient))
            or _rule_entry(RULE_UNSUPPORTED)
        )
        rule = str(params.get("rule", RULE_UNSUPPORTED))
        measured = y_t[:, idx]
        declared = y_p[:, idx]

        if rule == RULE_LEGACY_PIECEWISE:
            tol = _legacy_tolerance_from_measured(measured, params)
            cols.append((np.abs(declared - measured) <= tol).astype(np.float32))
            continue

        bounds = _declared_bounds(declared, params)
        if bounds is None:
            cols.append(np.full(measured.shape, np.nan, dtype=np.float32))
            continue

        lower, upper = bounds
        cols.append(((measured >= lower) & (measured <= upper)).astype(np.float32))

    return np.vstack(cols).T.astype(np.float32)


def eu_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nutrient_names: Sequence[str],
    config: Mapping[str, Mapping[str, object]],
) -> Dict[str, float]:
    """Compute per-nutrient and mean EU accuracy for supported nutrients."""
    within = compliance_matrix(y_true, y_pred, nutrient_names, config)
    out: Dict[str, float] = {}
    for idx, nutrient in enumerate(nutrient_names):
        col = within[:, idx]
        finite = np.isfinite(col)
        if not finite.any():
            out[f"eu_acc_{nutrient}"] = float("nan")
            continue
        out[f"eu_acc_{nutrient}"] = float(np.mean(col[finite]))

    finite_mask = np.isfinite(within)
    if finite_mask.any():
        out["eu_acc_mean"] = float(np.mean(within[finite_mask]))
        supported_rows = finite_mask.any(axis=1)
        all_targets = np.zeros(int(np.sum(supported_rows)), dtype=bool)
        row_idx = 0
        for row in range(within.shape[0]):
            row_finite = finite_mask[row]
            if not row_finite.any():
                continue
            all_targets[row_idx] = bool(np.all(within[row, row_finite] > 0.5))
            row_idx += 1
        out["eu_acc_all_targets"] = (
            float(np.mean(all_targets)) if row_idx else float("nan")
        )
    else:
        out["eu_acc_mean"] = float("nan")
        out["eu_acc_all_targets"] = float("nan")
    return out
