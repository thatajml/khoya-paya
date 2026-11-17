"""Loads params.yaml. Set KHOYA_PARAMS to point at a different file (tests do this)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=4)
def load_params(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("KHOYA_PARAMS", ROOT / "params.yaml")).resolve()
    with open(p, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params["_root"] = str(p.parent)
    return params


def resolve(params: dict, rel: str | os.PathLike) -> Path:
    """Paths in params.yaml are relative to the params file's folder."""
    path = Path(rel)
    return path if path.is_absolute() else Path(params["_root"]) / path
