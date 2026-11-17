"""The searchable gallery of found items.

For a few thousand items, a brute-force matrix product in NumPy is exact and
takes well under a millisecond, so we don't need FAISS yet. Swap it in when the
gallery reaches the hundreds of thousands (see README → Roadmap).

Scoring
-------
Text→image cosine similarities (~0.2–0.35) and image→image similarities
(~0.5–0.95) live on different scales, so each modality's scores are turned
into z-scores across the whole gallery ("how unusually similar is this item?")
before being averaged. Train/station matches then add a small boost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np


@dataclass
class Hit:
    item: dict
    z: float
    strength: str
    reasons: list[str] = field(default_factory=list)


def _zscores(sims: np.ndarray) -> np.ndarray:
    return (sims - sims.mean()) / (sims.std() + 1e-6)


class GalleryIndex:
    def __init__(self, embeddings: np.ndarray, items: list[dict], manifest: dict | None = None):
        assert len(embeddings) == len(items), "embeddings and items are out of sync"
        self.embeddings = embeddings.astype(np.float32)
        self.items = items
        self.manifest = manifest or {}
        self._found = np.array([date.fromisoformat(it["found_date"]) for it in items]) if items else np.array([])

    # ---------- persistence ----------
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.embeddings)
        (out / "items.json").write_text(json.dumps(self.items, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> "GalleryIndex":
        d = Path(index_dir)
        emb = np.load(d / "embeddings.npy")
        items = json.loads((d / "items.json").read_text(encoding="utf-8"))
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        return cls(emb, items, manifest)

    # ---------- lookups ----------
    @property
    def stations(self) -> list[str]:
        return sorted({it["station"] for it in self.items})

    @property
    def trains(self) -> list[dict]:
        seen = {it["train_no"]: it.get("train_name", "") for it in self.items if it.get("train_no")}
        return [{"train_no": k, "train_name": v} for k, v in sorted(seen.items())]

    def get(self, item_id: str) -> dict | None:
        return next((it for it in self.items if it["item_id"] == item_id), None)

    # ---------- search ----------
    def search(
        self,
        text_emb: np.ndarray | None = None,
        image_emb: np.ndarray | None = None,
        *,
        train_no: str | None = None,
        station: str | None = None,
        lost_date: date | None = None,
        top_k: int = 8,
        train_boost: float = 0.6,
        station_boost: float = 0.4,
        max_days_after_loss: int = 30,
        strong_z: float = 3.0,
        possible_z: float = 2.0,
    ) -> list[Hit]:
        if text_emb is None and image_emb is None:
            raise ValueError("Need a text description, a photo, or both.")
        if not self.items:
            return []

        parts = []
        if text_emb is not None:
            parts.append(_zscores(self.embeddings @ text_emb))
        if image_emb is not None:
            parts.append(_zscores(self.embeddings @ image_emb))
        score = np.mean(parts, axis=0)

        reasons: list[list[str]] = [[] for _ in self.items]
        if train_no:
            for i, it in enumerate(self.items):
                if it.get("train_no") == train_no:
                    score[i] += train_boost
                    reasons[i].append("same train")
        if station:
            for i, it in enumerate(self.items):
                if it.get("station", "").lower() == station.lower():
                    score[i] += station_boost
                    reasons[i].append("same station")

        mask = np.ones(len(self.items), dtype=bool)
        if lost_date is not None:
            mask = (self._found >= lost_date) & (self._found <= lost_date + timedelta(days=max_days_after_loss))

        order = [i for i in np.argsort(-score) if mask[i]][:top_k]
        hits = []
        for i in order:
            z = float(score[i])
            strength = "strong" if z >= strong_z else "possible" if z >= possible_z else "weak"
            hits.append(Hit(item=self.items[i], z=round(z, 3), strength=strength, reasons=reasons[i]))
        return hits
