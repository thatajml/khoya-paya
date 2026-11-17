"""Khoya-Paya serving API.

GET  /                  web UI
GET  /health            liveness + which model/index version is serving
GET  /api/meta          trains and stations for the filter dropdowns
POST /api/search        multipart: description, photo, train_no, station, lost_date
POST /api/feedback      {"query_id", "item_id", "rank", "action"} — "this is mine" clicks
GET  /images/{item_id}  gallery photo
GET  /metrics           Prometheus metrics
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from khoya import __version__
from khoya.config import load_params, resolve
from khoya.loader import load_encoder, load_index, search_kwargs
from khoya.query import parse_query

STATIC = Path(__file__).parent / "static"

# ---------- structured logging: one JSON object per line -> Vercel runtime logs ----------
log = logging.getLogger("khoya")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_handler)
log.propagate = False


def log_event(event: str, **fields) -> None:
    log.info(json.dumps({"event": event, "ts": round(time.time(), 3), **fields}, default=str))


# ---------- metrics ----------
SEARCHES = Counter("khoya_search_requests_total", "Search requests", ["mode"])
SEARCH_ERRORS = Counter("khoya_search_errors_total", "Rejected or failed searches", ["reason"])
LATENCY = Histogram("khoya_search_latency_seconds", "End-to-end search latency",
                    buckets=(0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2))
TOP_Z = Histogram("khoya_top_match_zscore", "z-score of the best match (drift signal: falling = worse matches)",
                  buckets=(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5))
FEEDBACK = Counter("khoya_feedback_total", "User feedback on results", ["action"])
CLICK_RANK = Histogram("khoya_claimed_rank", "Rank of the result users said was theirs",
                       buckets=(1, 2, 3, 5, 8))


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    params = load_params()
    app.state.params = params
    app.state.encoder = load_encoder(params)
    app.state.index = load_index(params)
    app.state.gallery = resolve(params, params["data"]["gallery_dir"])
    log_event("startup", load_ms=round(1000 * (time.perf_counter() - t0)),
              n_items=len(app.state.index.items), model=app.state.index.manifest.get("model"))
    yield


app = FastAPI(title="Khoya-Paya: railway lost & found matcher", version=__version__, lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health(request: Request) -> dict:
    idx = request.app.state.index
    return {"status": "ok", "version": __version__, "items": len(idx.items),
            "index_built_at": idx.manifest.get("built_at"), "model": idx.manifest.get("model")}


@app.get("/api/meta")
def meta(request: Request) -> dict:
    idx = request.app.state.index
    return {"stations": idx.stations, "trains": idx.trains, "items": len(idx.items),
            "categories": sorted({it["category"] for it in idx.items})}


@app.post("/api/search")
async def search(
    request: Request,
    description: str | None = Form(None, max_length=500),
    photo: UploadFile | None = File(None),
    train_no: str | None = Form(None),
    station: str | None = Form(None),
    lost_date: str | None = Form(None),
    top_k: int | None = Form(None),
) -> dict:
    t0 = time.perf_counter()
    st = request.app.state
    params = st.params
    query_id = uuid.uuid4().hex[:12]

    parsed = parse_query(description, st.index.stations)
    train_no = (train_no or "").strip() or parsed.train_no
    station = (station or "").strip() or parsed.station

    lost = None
    if lost_date:
        try:
            lost = date.fromisoformat(lost_date)
        except ValueError:
            SEARCH_ERRORS.labels("bad_date").inc()
            raise HTTPException(400, "lost_date must be YYYY-MM-DD")

    image_emb = None
    if photo is not None and photo.filename:
        limit = int(params["serving"]["max_upload_mb"] * 1024 * 1024)
        data = await photo.read(limit + 1)
        if len(data) > limit:
            SEARCH_ERRORS.labels("too_large").inc()
            raise HTTPException(413, f"Photo must be under {params['serving']['max_upload_mb']} MB")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError):
            SEARCH_ERRORS.labels("bad_image").inc()
            raise HTTPException(400, "Could not read that photo. Please upload a JPG or PNG.")
        # Model inference is CPU-bound: run it in a worker thread so the event loop stays responsive.
        image_emb = await run_in_threadpool(st.encoder.encode_image, img)

    text_emb = await run_in_threadpool(st.encoder.encode_text, parsed.clip_text) if parsed.clip_text else None
    if text_emb is None and image_emb is None:
        SEARCH_ERRORS.labels("empty").inc()
        raise HTTPException(400, "Describe the item or upload a photo of it.")

    k = max(1, min(top_k or params["search"]["top_k"], 24))
    hits = st.index.search(text_emb=text_emb, image_emb=image_emb, train_no=train_no, station=station,
                           lost_date=lost, top_k=k, **search_kwargs(params))

    mode = "text+image" if (text_emb is not None and image_emb is not None) else "image" if image_emb is not None else "text"
    elapsed = time.perf_counter() - t0
    SEARCHES.labels(mode).inc()
    LATENCY.observe(elapsed)
    if hits:
        TOP_Z.observe(hits[0].z)
    # Log a hash, not the passenger's raw words: enough to spot repeats, not to read them.
    log_event("search", query_id=query_id, mode=mode,
              text_sha=hashlib.sha256((description or "").encode()).hexdigest()[:10] if description else None,
              text_len=len(description or ""), train_filter=bool(train_no), station_filter=bool(station),
              date_filter=bool(lost), n_results=len(hits), top_z=hits[0].z if hits else None,
              top_item=hits[0].item["item_id"] if hits else None, latency_ms=round(1000 * elapsed, 1))

    return {
        "query_id": query_id,
        "understood": {"looking_for": parsed.clip_text, "train_no": train_no, "station": station,
                       "lost_date": lost.isoformat() if lost else None, "used_photo": image_emb is not None},
        "latency_ms": round(1000 * elapsed, 1),
        "results": [
            {"rank": r + 1, "item_id": h.item["item_id"], "strength": h.strength, "score": h.z,
             "reasons": h.reasons, "category": h.item["category"], "station": h.item["station"],
             "train_no": h.item["train_no"], "train_name": h.item["train_name"], "coach": h.item.get("coach"),
             "found_date": h.item["found_date"], "note": h.item["officer_note"],
             "image_url": f"/images/{h.item['item_id']}",
             "credit": {"license": h.item.get("license"), "author": h.item.get("author"),
                        "source_url": h.item.get("source_url")}}
            for r, h in enumerate(hits)
        ],
    }


class Feedback(BaseModel):
    query_id: str = Field(max_length=32)
    item_id: str = Field(max_length=32)
    rank: int = Field(ge=1, le=50)
    action: str = Field(default="this_is_mine", pattern="^(this_is_mine|not_mine)$")


@app.post("/api/feedback")
def feedback(fb: Feedback, request: Request) -> dict:
    if request.app.state.index.get(fb.item_id) is None:
        raise HTTPException(404, "Unknown item")
    FEEDBACK.labels(fb.action).inc()
    if fb.action == "this_is_mine":
        CLICK_RANK.observe(fb.rank)
    log_event("feedback", **fb.model_dump())
    return {"ok": True}


@app.get("/images/{item_id}")
def image(item_id: str, request: Request) -> FileResponse:
    item = request.app.state.index.get(item_id)
    if item is None:
        raise HTTPException(404, "Unknown item")
    return FileResponse(request.app.state.gallery / item["image"], media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
