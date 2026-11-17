"""Stage 0 (run once, locally) — download CC-licensed photos of commonly lost
items from Wikimedia Commons to act as our "found items" demo gallery.

    python -m pipeline.fetch_seed_images --per-category 8

Writes   data/raw/commons/<category>/<file>.jpg
and      data/seed_sources.csv  (title, licence, author, source page — attribution matters!)

Search results are noisy (drawings, shop shelves, people). After it finishes,
open data/raw/commons/ and DELETE anything that doesn't look like a single lost
item. That manual curation step is real data work — mention it in interviews.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path

from khoya.config import load_params, resolve

API = "https://commons.wikimedia.org/w/api.php"
CONTACT = os.environ.get("KHOYA_CONTACT", "student project; contact via GitHub")
HEADERS = {"User-Agent": f"KhoyaPaya/0.1 (lost-and-found demo; {CONTACT})"}

CATEGORIES = {
    "backpack": ["backpack", "rucksack bag"],
    "handbag": ["handbag", "leather purse bag"],
    "suitcase": ["suitcase", "trolley luggage"],
    "umbrella": ["umbrella", "folded umbrella"],
    "mobile phone": ["smartphone", "mobile phone"],
    "wallet": ["leather wallet"],
    "water bottle": ["water bottle"],
    "laptop": ["laptop computer"],
    "spectacles": ["eyeglasses", "spectacles"],
    "headphones": ["headphones", "earphones"],
    "keys": ["bunch of keys", "keychain"],
    "wristwatch": ["wristwatch"],
    "lunch box": ["tiffin box", "lunch box"],
    "shoes": ["pair of shoes", "sandals"],
    "jacket": ["jacket"],
    "teddy bear": ["teddy bear"],
}
BAD_TITLE = re.compile(
    r"\b(logo|icons?|diagram|drawing|sketch|map|patent|poster|advert\w*|chart|svg|scheme|plan|cartoon|clipart)\b", re.I
)


def api_search(term: str, limit: int) -> list[dict]:
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}", "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 640,
        "iiextmetadatafilter": "LicenseShortName|Artist",
    }
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    return pages


def clean_html(s: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()[:120]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=8)
    args = ap.parse_args()

    params = load_params()
    raw_root = resolve(params, params["data"]["raw_dir"]) / "commons"
    sources_csv = resolve(params, params["data"]["sources_csv"])
    rows = []

    for category, terms in CATEGORIES.items():
        out_dir = raw_root / category.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        got = 0
        for term in terms:
            if got >= args.per_category:
                break
            try:
                pages = api_search(term, limit=args.per_category * 3)
            except Exception as e:  # network hiccups shouldn't kill the whole run
                print(f"  ! search failed for {term!r}: {e}")
                continue
            for page in pages:
                if got >= args.per_category:
                    break
                info = (page.get("imageinfo") or [{}])[0]
                title = page.get("title", "")
                if info.get("mime") not in ("image/jpeg", "image/png") or BAD_TITLE.search(title):
                    continue
                if min(info.get("width", 0), info.get("height", 0)) < 400:
                    continue
                slug = re.sub(r"[^a-z0-9]+", "-", title.lower().removeprefix("file:"))[:60].strip("-")
                dest = out_dir / f"{slug}.jpg"
                if not dest.exists():
                    try:
                        req = urllib.request.Request(info["thumburl"], headers=HEADERS)
                        with urllib.request.urlopen(req, timeout=60) as r:
                            dest.write_bytes(r.read())
                    except Exception as e:
                        print(f"  ! download failed: {title}: {e}")
                        continue
                    time.sleep(0.5)  # be polite to Wikimedia
                meta = info.get("extmetadata", {})
                rows.append({
                    "raw_path": dest.relative_to(raw_root.parent).as_posix(),
                    "category": category,
                    "title": title,
                    "license": clean_html(meta.get("LicenseShortName", {}).get("value", "")),
                    "author": clean_html(meta.get("Artist", {}).get("value", "")),
                    "source_url": info.get("descriptionurl", ""),
                })
                got += 1
        print(f"{category:14s} {got} images")

    sources_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(sources_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["raw_path"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} images in {raw_root}. Now curate them, then run: python -m pipeline.ingest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
