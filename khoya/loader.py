"""Builds the encoder and index from params.yaml, so the pipeline and the app load them identically."""

from __future__ import annotations

from .clip_onnx import ClipEncoder
from .config import resolve
from .index import GalleryIndex


def load_encoder(params: dict) -> ClipEncoder:
    return ClipEncoder(
        resolve(params, params["model"]["dir"]),
        variant=params["model"]["variant"],
        threads=params.get("serving", {}).get("intra_op_threads", 1),
    )


def load_index(params: dict) -> GalleryIndex:
    return GalleryIndex.load(resolve(params, params["index"]["dir"]))


def search_kwargs(params: dict) -> dict:
    s = params["search"]
    return {k: s[k] for k in ("train_boost", "station_boost", "max_days_after_loss", "strong_z", "possible_z")}
