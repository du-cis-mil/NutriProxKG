"""Target-nutrient -> compound-vocabulary alias map (leakage control).

The Task-2 labels (``data/eval/task2_nutrition*.csv``) are FooDB nutrient values.
FooDB *also* exposes those same nutrients as INGREDIENT_COMPOUND entries
("Calcium", "Proteins", "Carbohydrate", "Retinol", "Ascorbic acid", ...).  If a
compound node is literally the target nutrient (or a clearly-equivalent chemical
form of it), then ``has_compound`` / ``similar_to`` carry the *presence* of the
prediction target into the graph.  This module identifies those compound nodes
so they can be masked for the leakage-control experiments.

Two match tiers:
  - ``matched``  : the compound IS the target nutrient or a clear alias / named
                   vitamer form.  These are *masked* in the nutrient-masked
                   variant.  Macros & minerals use exact-name matches only; the
                   named vitamins (whose label column names the vitamer) also
                   match their vitamer family by a curated stem.
  - ``possible`` : element-bearing salts / derivatives that *contain* the
                   nutrient name but are distinct chemicals (e.g. "Calcium
                   gluconate", "Sodium chloride").  Reported for transparency,
                   masked only under ``aggressive=True``.

This is the single source of truth shared by ``analysis/leakage_check.py`` and
``e2e/build_similar_to_variants.py`` / ``e2e/leakage_control.py``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# Task-2 target columns per track (see e2e/config.py TRACKS).
NUTRI3_TARGETS = ["carbs_g", "fat_g", "protein_g"]
NUTRI19_TARGETS = [
    "carbs_g",
    "fat_g",
    "protein_g",
    "calcium_g",
    "copper_g",
    "iodine_g",
    "iron_g",
    "manganese_g",
    "phosphorus_g",
    "potassium_g",
    "selenium_g",
    "sodium_g",
    "vitamin_a_retinol_g",
    "vitamin_b1_thiamine_g",
    "vitamin_b2_riboflavin_g",
    "vitamin_b9_folate_g",
    "vitamin_c_g",
    "vitamin_e_tocopherol_g",
    "zinc_g",
]

# Per target: exact normalised names (the nutrient itself + clear synonyms) and,
# for the named vitamins, a curated vitamer-family stem matched as a substring of
# the normalised compound name.  Minerals/macros use exact only (element-substring
# matching would wrongly capture additives like "sodium benzoate").
NUTRIENT_SPEC: Dict[str, Dict[str, List[str]]] = {
    "carbs_g": {
        "exact": [
            "carbohydrate",
            "carbohydrates",
            "total carbohydrate",
            "carbohydrate total",
            "carbohydrate by difference",
        ]
    },
    "fat_g": {
        "exact": [
            "fat",
            "fats",
            "total fat",
            "fat total",
            "total lipid",
            "total lipids",
            "total lipid fat",
        ]
    },
    "protein_g": {"exact": ["protein", "proteins"]},
    # Minerals: exact name plus the USDA "Element, Symbol" form FoodKG uses
    # (e.g. "Calcium, Ca" -> "calcium ca").  These extra exacts never match the
    # FooDB chemical vocabulary (proposed/ICS), where the bare "calcium" already
    # matches, so they only close the FoodKG mineral gap.  element-substring stays
    # exact-token-only (so additives like "sodium benzoate" are not masked).
    "calcium_g": {"exact": ["calcium", "calcium ca"], "element": ["calcium"]},
    "copper_g": {"exact": ["copper", "copper cu"], "element": ["copper"]},
    "iodine_g": {"exact": ["iodine", "iodine i"], "element": ["iodine"]},
    "iron_g": {"exact": ["iron", "iron fe"], "element": ["iron"]},
    "manganese_g": {"exact": ["manganese", "manganese mn"], "element": ["manganese"]},
    "phosphorus_g": {
        "exact": ["phosphorus", "phosphorus p"],
        "element": ["phosphorus"],
    },
    "potassium_g": {"exact": ["potassium", "potassium k"], "element": ["potassium"]},
    "selenium_g": {"exact": ["selenium", "selenium se"], "element": ["selenium"]},
    "sodium_g": {"exact": ["sodium", "sodium na"], "element": ["sodium"]},
    "zinc_g": {"exact": ["zinc", "zinc zn"], "element": ["zinc"]},
    # Vitamins: the label column names the vitamer, and FooDB stores several
    # chemical forms of the SAME vitamer as distinct compounds (e.g. the aldehyde
    # 'Retinal', the ester 'retinyl palmitate', the B2 cofactors 'FMN'/'FAD', the
    # folate 'pteroyl-glutamate' forms, the oxidised vitamin-C 'dehydroascorbate').
    # We mask the whole named-vitamer family.  Stems are curated to avoid false
    # positives (NB: 'flavin' would wrongly hit tea 'theaflavin' polyphenols, so
    # the B2 family uses the specific cofactor abbreviations 'fmn'/'fad' instead).
    "vitamin_a_retinol_g": {
        "exact": ["retinol", "vitamin a"],
        "family": ["retinol", "retinal", "retinaldehyde", "retinyl", "retinoic"],
    },
    "vitamin_b1_thiamine_g": {
        "exact": ["thiamine", "thiamin", "vitamin b1"],
        "family": ["thiamin", "cocarboxylase"],
    },
    # B2 cofactors FMN/FAD are matched by EXACT bare name (substring 'fad' would
    # wrongly hit 'Sulfadiazine'/'Sulfadimidine'; substring 'fmn' is safe but kept
    # exact for symmetry).
    "vitamin_b2_riboflavin_g": {
        "exact": [
            "riboflavin",
            "riboflavine",
            "vitamin b2",
            "fmn",
            "fmnh",
            "fad",
            "fadh2",
        ],
        "family": ["riboflavin"],
    },
    "vitamin_b9_folate_g": {
        "exact": ["folate", "folic acid", "vitamin b9"],
        "family": ["folate", "folic", "pteroyl"],
    },
    "vitamin_c_g": {
        "exact": ["vitamin c", "ascorbic acid", "l ascorbic acid"],
        "family": ["ascorb"],
    },
    "vitamin_e_tocopherol_g": {
        "exact": ["tocopherol", "vitamin e", "alpha tocopherol"],
        "family": ["tocopherol", "tocotrienol"],
    },
}

# Substrings that look like a vitamer family but are unrelated chemicals.
FAMILY_DENYLIST = ["pinifolic"]  # e.g. "Dehydropinifolic acid" contains "folic"


def normalize_compound(name: str) -> str:
    """lowercase + punctuation->space + whitespace-collapse."""
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def targets_for_track(track: str) -> List[str]:
    return list(NUTRI3_TARGETS) if track in ("n3", "nutri3") else list(NUTRI19_TARGETS)


def _classify(norm: str, spec: Dict[str, List[str]]) -> str:
    """Return 'exact' | 'family' | '' for one compound vs one target spec."""
    if norm in set(spec.get("exact", [])):
        return "exact"
    if any(d in norm for d in FAMILY_DENYLIST):
        return ""
    for stem in spec.get("family", []):
        if stem in norm:
            return "family"
    return ""


def _is_possible(norm: str, spec: Dict[str, List[str]]) -> bool:
    """Element name appears as a whitespace-bounded token but is not an exact match."""
    toks = set(norm.split())
    return any(el in toks for el in spec.get("element", []))


def compute_matches(
    cmp_names: List[str],
    targets: List[str],
) -> Tuple[List[Dict], List[Dict]]:
    """Return (matched, possible) records for the given compound vocabulary.

    Each record: {cmp_idx, cmp_name, norm_name, nutrient, match_type}.
    matched   -> exact / family hits (masked in the nutrient-masked variant).
    possible  -> element-bearing salts/derivatives (reported, not masked).
    """
    matched: List[Dict] = []
    possible: List[Dict] = []
    norms = [normalize_compound(c) for c in cmp_names]
    for i, (name, norm) in enumerate(zip(cmp_names, norms)):
        hit_target, hit_type = None, None
        for tgt in targets:
            mt = _classify(norm, NUTRIENT_SPEC[tgt])
            if mt:
                hit_target, hit_type = tgt, mt
                break
        if hit_target is not None:
            matched.append(
                {
                    "cmp_idx": i,
                    "cmp_name": name,
                    "norm_name": norm,
                    "nutrient": hit_target,
                    "match_type": hit_type,
                }
            )
            continue
        for tgt in targets:
            if _is_possible(norm, NUTRIENT_SPEC[tgt]):
                possible.append(
                    {
                        "cmp_idx": i,
                        "cmp_name": name,
                        "norm_name": norm,
                        "nutrient": tgt,
                        "match_type": "possible_element_salt",
                    }
                )
                break
    return matched, possible


def masked_compound_ids(
    cmp_names: List[str], track: str, aggressive: bool = False
) -> Set[int]:
    """Compound indices to mask for ``track``.

    aggressive=False : exact + named-vitamer-family matches (default).
    aggressive=True  : also mask element-bearing salts/derivatives (sensitivity).
    """
    targets = targets_for_track(track)
    matched, possible = compute_matches(cmp_names, targets)
    ids = {m["cmp_idx"] for m in matched}
    if aggressive:
        ids |= {p["cmp_idx"] for p in possible}
    return ids
