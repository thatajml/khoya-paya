"""Stage 3 — measure retrieval quality and latency; optionally fail the build.

    python -m pipeline.evaluate            # report only
    python -m pipeline.evaluate --gate     # exit 1 if any metric is below its gate (CI / Docker build)
    python -m pipeline.evaluate --mlflow   # also log the run to MLflow (pip install mlflow)

Three test sets, none of which need a human to label anything except the last:

1. image -> image ("I have an old photo of my bag")
   Each gallery photo is distorted (crop, rotate, lighting, blur, JPEG
   compression) to imitate a passenger's own photo; we check that the original
   item is ranked in the top 1 / top 5.  -> img2img_recall_at_1/5, img2img_mrr

2. text -> category ("blue umbrella")
   For every category, query "a photo of a <category>" and check what fraction
   of the top 5 are that category.       -> text_category_precision_at_5

3. human queries (optional, the most realistic one)
   data/eval_queries.csv with columns: query,expected_item_id
   Write these yourself, the way a worried passenger would.  -> human_recall_at_5
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import statistics
import sys
import time
from collections import defaultdict

from PIL import Image, ImageEnhance, ImageFilter

from khoya.config import load_params, resolve
from khoya.loader import load_encoder, load_index, search_kwargs
from khoya.query import parse_query


def passenger_photo(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a different photo of the same object."""
    img = img.convert("RGB")
    w, h = img.size
    s = rng.uniform(0.75, 0.92)
    cw, ch = int(w * s), int(h * s)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x, y, x + cw, y + ch))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rng.uniform(-12, 12), resample=Image.BICUBIC, expand=False, fillcolor=(235, 235, 235))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(40, 75))
    return Image.open(io.BytesIO(buf.getvalue()))


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="exit non-zero if a metric is below its gate")
    ap.add_argument("--mlflow", action="store_true", help="log params and metrics to MLflow")
    args = ap.parse_args()

    params = load_params()
    rng = random.Random(params["eval"]["seed"])
    encoder, index = load_encoder(params), load_index(params)
    gallery = resolve(params, params["data"]["gallery_dir"])
    kw = search_kwargs(params)
    metrics: dict[str, float] = {}

    # 1. image -> image
    ranks, img_lat = [], []
    for it in index.items:
        query_img = passenger_photo(Image.open(gallery / it["image"]), rng)
        t0 = time.perf_counter()
        emb = encoder.encode_image(query_img)
        hits = index.search(image_emb=emb, top_k=len(index.items), **kw)
        img_lat.append(1000 * (time.perf_counter() - t0))
        ids = [h.item["item_id"] for h in hits]
        ranks.append(ids.index(it["item_id"]) + 1)
    n = len(ranks)
    metrics["img2img_recall_at_1"] = sum(r <= 1 for r in ranks) / n
    metrics["img2img_recall_at_5"] = sum(r <= 5 for r in ranks) / n
    metrics["img2img_mrr"] = sum(1 / r for r in ranks) / n

    # 2. text -> category
    by_cat = defaultdict(int)
    for it in index.items:
        by_cat[it["category"]] += 1
    precisions, txt_lat = {}, []
    for cat, count in sorted(by_cat.items()):
        if count < 3:
            continue
        k = min(5, count)
        article = "an" if cat[0] in "aeiou" else "a"
        t0 = time.perf_counter()
        emb = encoder.encode_text(f"a photo of {article} {cat}")
        hits = index.search(text_emb=emb, top_k=k, **kw)
        txt_lat.append(1000 * (time.perf_counter() - t0))
        precisions[cat] = sum(h.item["category"] == cat for h in hits) / k
    if precisions:
        metrics["text_category_precision_at_5"] = statistics.mean(precisions.values())

    # 3. human-written queries (optional)
    human_csv = resolve(params, params["data"]["human_eval_csv"])
    if human_csv.exists():
        with open(human_csv, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("query") and r.get("expected_item_id")]
        hits_at_5 = 0
        for r in rows:
            pq = parse_query(r["query"], index.stations)
            if not pq.clip_text:
                continue
            hits = index.search(text_emb=encoder.encode_text(pq.clip_text), train_no=pq.train_no,
                                station=pq.station, top_k=5, **kw)
            hits_at_5 += r["expected_item_id"] in [h.item["item_id"] for h in hits]
        if rows:
            metrics["human_recall_at_5"] = hits_at_5 / len(rows)

    metrics["latency_image_query_p50_ms"] = pct(img_lat, 0.5)
    metrics["latency_image_query_p95_ms"] = pct(img_lat, 0.95)
    if txt_lat:
        metrics["latency_text_query_p50_ms"] = pct(txt_lat, 0.5)
    metrics = {k: round(v, 4) for k, v in metrics.items()}

    # report
    gates = params["eval"]["gates"]
    failed = []
    print(f"\nEvaluation on {n} items  (model variant: {params['model']['variant']})")
    print("-" * 58)
    for k, v in metrics.items():
        gate = gates.get(k)
        status = ""
        if gate is not None:
            ok = v >= gate
            status = f"gate {gate:.2f}  {'PASS' if ok else 'FAIL'}"
            if not ok:
                failed.append(k)
        print(f"{k:34s} {v:9.3f}   {status}")
    if precisions:
        worst = sorted(precisions.items(), key=lambda kv: kv[1])[:3]
        print("weakest text categories:", ", ".join(f"{c} ({p:.2f})" for c, p in worst))

    out = resolve(params, params["eval"]["metrics_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "per_category_precision": precisions,
                               "index": index.manifest}, indent=2))
    print(f"\nWrote {out}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("khoya-paya-retrieval")
        with mlflow.start_run():
            mlflow.log_params({"variant": params["model"]["variant"], "revision": params["model"]["revision"][:12],
                               "n_items": n, **{f"search.{k}": v for k, v in kw.items()}})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out))
        print("Logged run to MLflow (run `mlflow ui` to compare runs)")

    if args.gate and failed:
        print(f"\nQUALITY GATE FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
