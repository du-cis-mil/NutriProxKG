"""Text normalization for ingredient names."""

from __future__ import annotations

import re
from typing import List


_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_SEP_RE = re.compile(r"[\s\-/_,;:(){}\[\]]+")


def normalize_name(text: str) -> str:
    """Normalize ingredient text for robust key matching."""
    value = str(text or "").strip().lower()
    value = value.replace("&", " and ")
    value = _PUNCT_RE.sub(" ", value)
    value = " ".join(value.split())
    return value


def tokenize_name(text: str) -> List[str]:
    """Tokenize ingredient name by separators and punctuation."""
    base = str(text or "").strip().lower()
    parts = [p for p in _SEP_RE.split(base) if p]
    tokens: List[str] = []
    for token in parts:
        clean = normalize_name(token)
        if clean:
            tokens.extend(clean.split())
    return [t for t in tokens if t]
