"""Stage 2 — embed every gallery photo with CLIP and save the search index.

    python -m pipeline.build_index

The manifest ties this index to the exact model weights and data it was built
from (lineage). The app shows it at /health, so you can always answer
"which model and data is production running?"
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone

from PIL import Image

from khoya.config import load_params, resolve
from khoya.index import GalleryIndex
from khoya.loader import load_encoder


def main() -> int:
    params = load_params()
    d = params["data"]
    gallery = resolve(params, d["gallery_dir"])
    items_csv = resolve(params, d["items_csv"])
    with open(items_csv, encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    if not items:
        print("No items in", items_csv)
        return 1

    encoder = load_encoder(params)
    t0 = time.perf_counter()
    images = [Image.open(gallery / it["image"]) for it in items]
    emb = encoder.encode_images(images)
    secs = time.perf_counter() - t0

    model_manifest_path = resolve(params, params["model"]["dir"]) / "model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text()) if model_manifest_path.exists() else {}
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_items": len(items),
        "dim": int(emb.shape[1]),
        "items_csv_sha256": hashlib.sha256(items_csv.read_bytes()).hexdigest(),
        "model": {k: model_manifest.get(k) for k in ("hf_repo", "revision", "variant")},
        "model_files_sha256": {k: v["sha256"][:12] for k, v in model_manifest.get("files", {}).items()},
    }
    GalleryIndex(emb, items, manifest).save(resolve(params, params["index"]["dir"]))
    print(f"Indexed {len(items)} items (dim={emb.shape[1]}) in {secs:.1f}s "
          f"({1000 * secs / len(items):.0f} ms/image)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
