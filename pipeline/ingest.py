"""Stage 1 — turn raw photos into the clean gallery + found_items.csv.

    python -m pipeline.ingest

Input layout:   data/raw/<source>/<category>/<photo>
                  e.g. data/raw/commons/backpack/...   (from fetch_seed_images)
                       data/raw/own/umbrella/IMG_2041.jpg   (your own phone photos)

For every photo it:
  1. fixes phone rotation (EXIF) and converts to RGB
  2. blurs faces (privacy — found-item photos often catch bystanders)
  3. resizes to max 512 px and re-saves WITHOUT EXIF (drops GPS location)
  4. gives it a stable ID derived from its path (re-runs don't reshuffle IDs)
  5. generates demo "found by RPF" metadata: train, station, coach, date

Rows you edit by hand in data/found_items.csv are kept on re-runs.
"""

from __future__ import annotations

import csv
import hashlib
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from khoya.config import load_params, resolve

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Real Western/Central Railway trains around Mumbai, with stations on their routes.
TRAINS = [
    ("12951", "Mumbai Rajdhani", ["Mumbai Central", "Borivali", "Surat", "Vadodara"]),
    ("12009", "Ahmedabad Shatabdi", ["Mumbai Central", "Borivali", "Surat", "Vadodara", "Ahmedabad"]),
    ("12933", "Karnavati Express", ["Mumbai Central", "Dadar", "Borivali", "Surat", "Ahmedabad"]),
    ("12921", "Flying Ranee", ["Mumbai Central", "Dadar", "Borivali", "Surat"]),
    ("12123", "Deccan Queen", ["Mumbai CSMT", "Dadar", "Thane", "Kalyan", "Lonavala", "Pune Junction"]),
    ("12125", "Pragati Express", ["Mumbai CSMT", "Dadar", "Thane", "Kalyan", "Lonavala", "Pune Junction"]),
    ("12127", "Pune Intercity", ["Mumbai CSMT", "Dadar", "Thane", "Kalyan", "Pune Junction"]),
]
COACHES = ["S2", "S5", "B1", "B3", "A1", "C2", "C4", "D3", "GS"]

FIELDS = [
    "item_id", "image", "category", "source", "station", "train_no", "train_name", "coach",
    "found_date", "officer_note", "license", "author", "source_url",
]


def stable_id(rel_path: str) -> str:
    return "FI-" + hashlib.sha1(rel_path.encode()).hexdigest()[:6].upper()


def blur_faces(img: Image.Image) -> tuple[Image.Image, int]:
    try:
        import cv2
    except ImportError:  # opencv is a dev dependency; skip gracefully if missing
        return img, 0
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    for (x, y, w, h) in faces:
        x, y, w, h = int(x), int(y), int(w), int(h)
        pad = int(0.15 * max(w, h))
        box = (max(0, x - pad), max(0, y - pad), min(img.width, x + w + pad), min(img.height, y + h + pad))
        region = img.crop(box).filter(ImageFilter.GaussianBlur(radius=max(8, w // 4)))
        img.paste(region, box)
    return img, len(faces)


def synth_metadata(item_id: str, category: str, ref: date) -> dict:
    rng = random.Random(item_id)  # deterministic per item
    train_no, train_name, stations = rng.choice(TRAINS)
    station = rng.choice(stations)
    coach = rng.choice(COACHES)
    found = ref - timedelta(days=rng.randint(0, 45))
    if rng.random() < 0.7:
        where = f"in coach {coach} of {train_no} {train_name}"
    else:
        where, coach = f"on platform {rng.randint(1, 8)}", ""
    note = f"{category.capitalize()} found {where} at {station}. Deposited at the Lost Property Office."
    return {"station": station, "train_no": train_no, "train_name": train_name, "coach": coach,
            "found_date": found.isoformat(), "officer_note": note}


def main() -> int:
    params = load_params()
    d = params["data"]
    raw_root = resolve(params, d["raw_dir"])
    gallery = resolve(params, d["gallery_dir"])
    items_csv = resolve(params, d["items_csv"])
    ref = date.fromisoformat(str(d["reference_date"]))
    gallery.mkdir(parents=True, exist_ok=True)

    sources = {}
    src_csv = resolve(params, d["sources_csv"])
    if src_csv.exists():
        with open(src_csv, encoding="utf-8") as f:
            sources = {r["raw_path"]: r for r in csv.DictReader(f)}
    existing = {}
    if items_csv.exists():
        with open(items_csv, encoding="utf-8") as f:
            existing = {r["item_id"]: r for r in csv.DictReader(f)}

    rows, faces_total = [], 0
    for path in sorted(raw_root.rglob("*")):
        if path.suffix.lower() not in IMG_EXT:
            continue
        rel = path.relative_to(raw_root).as_posix()  # <source>/<category>/<file>
        parts = rel.split("/")
        if len(parts) < 3:
            print(f"  skip {rel}: expected raw/<source>/<category>/<file>")
            continue
        source, category = parts[0], parts[1].replace("_", " ").lower()
        item_id = stable_id(rel)

        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as e:
            print(f"  skip {rel}: {e}")
            continue
        img.thumbnail((d["max_side"], d["max_side"]), Image.LANCZOS)
        if d.get("blur_faces", True):
            img, n = blur_faces(img)
            faces_total += n
        out_name = f"{item_id}.jpg"
        img.save(gallery / out_name, "JPEG", quality=88)  # PIL drops EXIF unless asked to keep it

        src = sources.get(rel, {})
        row = {"item_id": item_id, "image": out_name, "category": category, "source": source,
               "license": src.get("license", "own photo" if source == "own" else ""),
               "author": src.get("author", ""), "source_url": src.get("source_url", "")}
        if item_id in existing:  # keep hand edits
            row.update({k: existing[item_id][k] for k in
                        ("station", "train_no", "train_name", "coach", "found_date", "officer_note")})
        else:
            row.update(synth_metadata(item_id, category, ref))
        rows.append(row)

    keep = {r["image"] for r in rows}
    for stale in gallery.glob("*.jpg"):  # images whose raw photo was deleted during curation
        if stale.name not in keep:
            stale.unlink()

    with open(items_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    cats = sorted({r["category"] for r in rows})
    print(f"{len(rows)} items across {len(cats)} categories -> {items_csv} ({faces_total} faces blurred)")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
