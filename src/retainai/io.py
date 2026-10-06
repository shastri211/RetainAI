"""Deterministic file writers (stable bytes across platforms)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    """Serialise numpy scalars/arrays; anything else falls back to ``str``."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV with LF line endings and no index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
    return path


def write_json(payload: Any, path: str | Path) -> Path:
    """Write sorted-key, LF-terminated JSON (stable bytes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    return path
