"""Raw-data loading and dataset fingerprinting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV with every field as text.

    Reading everything as text keeps blank cells (e.g. ``TotalCharges == " "``) visible to
    validation instead of letting the parser silently coerce them.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def dataset_sha256(path: str | Path) -> str:
    """SHA-256 of the file content with line endings normalised to LF.

    Git on Windows (``autocrlf=true``) checks the CSV out with CRLF while the repository stores LF,
    so a raw-byte hash would differ between machines. Normalising makes the fingerprint portable.
    """
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()
