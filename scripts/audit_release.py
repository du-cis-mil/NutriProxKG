"""Inspect this release folder for accidental data, local paths and secrets.

This conservative heuristic is a review aid, not proof of absence of all secrets.
--clean-generated removes only disposable Python/build metadata, never research data.
"""

import argparse
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def audit(clean_generated=False):
    if clean_generated:
        for p in sorted(ROOT.rglob("*"), reverse=True):
            if p.is_dir() and (p.name == "__pycache__" or p.name.endswith(".egg-info")):
                shutil.rmtree(p)
    findings = []
    files = []
    patterns = {
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "local user or mounted path": re.compile(r'(?<![\w.])/(?:home|Users|mnt)/[^\s"\'<>]+'),
        "private IPv4": re.compile(
            r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"
        ),
    }
    prohibited = {
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".ckpt",
        ".npz",
        ".npy",
        ".csv",
        ".tsv",
        ".dmp",
        ".owl",
        ".pdf",
        ".zip",
        ".gz",
        ".log",
        ".pem",
        ".key",
    }
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_symlink():
            findings.append(f"{rel}: symlink")
            continue
        if p.is_dir():
            if p.name in {"__pycache__", "build", "dist"} or p.name.endswith(
                ".egg-info"
            ):
                findings.append(f"{rel}: generated directory")
            continue
        if not p.is_file():
            continue
        files.append(p)
        if p.name.startswith(".env") or p.suffix in prohibited:
            findings.append(f"{rel}: prohibited artifact type")
        if rel.parts[0] in {"data", "outputs"} and p.name != ".gitkeep":
            findings.append(f"{rel}: source/derived/output artifact")
        if p.stat().st_size > 1024 * 1024:
            findings.append(f"{rel}: exceeds 1 MiB")
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{rel}: binary file")
            continue
        for label, pattern in patterns.items():
            if pattern.search(content):
                findings.append(f"{rel}: possible {label}")
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        ".gitignore",
        "pyproject.toml",
        "requirements-cpu.lock",
        "docs/data.md",
        "docs/reproducibility.md",
        "docs/validation.md",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            findings.append(f"{name}: required file missing")
    if findings:
        print("\n".join(findings))
        return 1
    print(
        f"PASS: {len(files)} text/empty files; {sum(p.stat().st_size for p in files)} bytes; no flagged data, binaries, large files, symlinks, local paths or credential patterns."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-generated", action="store_true")
    args = parser.parse_args()
    sys.exit(audit(args.clean_generated))
