"""Stage 1 — download the pinned CLIP ONNX files and record their checksums.

Pinning a commit hash (not "main") means every build gets byte-identical
weights; the sha256 manifest proves it and becomes part of the index lineage.

    python -m pipeline.fetch_model
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

from khoya.clip_onnx import model_files
from khoya.config import load_params, resolve


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "khoya-paya/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    params = load_params()
    m = params["model"]
    out_dir = resolve(params, m["dir"])
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    manifest_path = out_dir / "model_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    files = {}
    for rel in model_files(m["variant"]).values():
        dest = out_dir / rel
        cached = old.get("revision") == m["revision"] and rel in old.get("files", {})
        if cached and dest.exists() and sha256(dest) == old["files"][rel]["sha256"]:
            print(f"  {rel}: cached")
        else:
            download(f"{endpoint}/{m['hf_repo']}/resolve/{m['revision']}/{rel}", dest)
        files[rel] = {"sha256": sha256(dest), "bytes": dest.stat().st_size}

    manifest = {"hf_repo": m["hf_repo"], "revision": m["revision"], "variant": m["variant"], "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total_mb = sum(f["bytes"] for f in files.values()) / 1e6
    print(f"Model ready in {out_dir} ({total_mb:.0f} MB, variant={m['variant']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
