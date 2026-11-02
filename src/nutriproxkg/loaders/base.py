"""Base utilities for dataset loaders."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional


class LoaderBase:
    """Base loader with helper methods for file discovery and parsing."""

    def __init__(
        self, raw_dir: str | Path, logger: Optional[logging.Logger] = None
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._all_files_cache: Optional[list[Path]] = None

    def find_file(self, candidates: Iterable[str]) -> Optional[Path]:
        """Find first matching file recursively from candidate names."""
        all_files = self._list_all_files()
        name_to_paths: Dict[str, list[Path]] = {}
        for p in all_files:
            name_to_paths.setdefault(p.name, []).append(p)

        for candidate in candidates:
            if candidate in name_to_paths:
                return sorted(name_to_paths[candidate])[0]
            lower_candidate = candidate.lower()
            hits_ci = [
                p
                for name, paths in name_to_paths.items()
                if name.lower() == lower_candidate
                for p in paths
            ]
            if hits_ci:
                return sorted(hits_ci)[0]
        return None

    def _iter_all_files(self) -> Generator[Path, None, None]:
        for dirpath, _, filenames in os.walk(self.raw_dir, followlinks=True):
            for filename in filenames:
                yield Path(dirpath) / filename

    def _list_all_files(self) -> list[Path]:
        if self._all_files_cache is None:
            self._all_files_cache = list(self._iter_all_files())
        return self._all_files_cache


def detect_delimiter(path: Path) -> str:
    """Infer delimiter for text table files."""
    if path.suffix.lower() in {".tsv", ".tab"}:
        return "\t"

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = "".join([f.readline() for _ in range(5)])

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        return ","


def iter_table_rows(path: Path) -> Generator[Dict[str, str], None, None]:
    """Yield rows from CSV/TSV with detected delimiter."""
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield {
                k.strip(): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
                if k is not None
            }


def iter_records(path: Path) -> Generator[Dict[str, object], None, None]:
    """Yield records from CSV/TSV/JSON/JSONL files."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".tab", ".txt"}:
        for row in iter_table_rows(path):
            yield row
        return

    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            if suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield rec
                return

            payload = json.load(f)
            if isinstance(payload, list):
                for rec in payload:
                    if isinstance(rec, dict):
                        yield rec
            elif isinstance(payload, dict):
                yield payload


def normalize_text(value: object) -> str:
    """Normalize text value."""
    return " ".join(str(value or "").strip().split())


def parse_float(value: object) -> Optional[float]:
    """Parse float safely."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None
