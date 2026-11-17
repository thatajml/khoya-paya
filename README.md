# Khoya-Paya · AI lost & found for Indian Railways

> *Khoya-paya* (खोया-पाया) = "lost and found".

Passengers lose thousands of bags, phones and umbrellas on trains every year. Under **Mission/Operation Amanat**, the RPF posts found items with photos on the Western Railway website, but passengers have to scroll through divisional galleries by hand. There is no search and no matching.

**Khoya-Paya** lets a passenger describe the item in their own words, or upload an old photo of it. It then ranks every found item by how likely it is to be theirs. The model is **CLIP**, which embeds images and text in the same vector space. Serving runs on **ONNX Runtime** inside **Docker** on **Vercel**, behind an automated quality gate.

```
"black backpack with a red keychain, lost on 12951 near Borivali"
        │
        ├─ query parser ──► train 12951 · station Borivali       (boost)
        └─ CLIP text enc ─► 512-d vector ─┐
                                          ├─► cosine sim vs gallery ─► z-score ─► ranked matches
   optional photo ─► CLIP image enc ──────┘
```

---

## Architecture

```mermaid
flowchart LR
  subgraph Offline pipeline — runs in CI and in the Docker build
    A[Wikimedia Commons<br/>+ your own photos] -->|fetch_seed_images| B[data/raw]
    B -->|ingest: EXIF fix, face blur,<br/>strip GPS, resize, stable IDs| C[data/gallery<br/>found_items.csv]
    M[HF Hub, pinned commit] -->|fetch_model + sha256 manifest| D[CLIP ONNX int8]
    C & D -->|build_index| E[embeddings.npy<br/>items.json + manifest]
    E -->|evaluate --gate| F{metrics ≥ gates?}
  end
  F -- no --> X[build fails, nothing deploys]
  F -- yes --> G[runtime image:<br/>FastAPI + ONNX Runtime]
  G --> V[Vercel Container Function]
  V --> L[JSON logs · /metrics · /health]
```

| Layer | Choice | Why |
|---|---|---|
| Model | `openai/clip-vit-base-patch32`, int8 ONNX export by Xenova | No training needed on day 1. About 150 MB; runs on 1 vCPU. |
| Inference | `onnxruntime`, no PyTorch at runtime | The image is ~4× smaller and cold starts are faster. |
| Search | NumPy brute force over L2-normalised vectors | Exact, and under 1 ms for a few thousand items. Swap in FAISS or pgvector at scale. |
| Ranking | Per-modality z-scores, averaged, plus train/station boosts | Text→image and image→image cosines are on different scales. |
| API | FastAPI | Typed, async, gives free `/docs`. |
| Packaging | Multi-stage Dockerfile | The build stage includes the quality gate; the runtime stage stays slim. |
| Hosting | Vercel Container Runtime (beta) | Deploys on every git push, scales to zero. |

---

## Repo map

```
khoya/                 core library shared by pipeline and app
  clip_onnx.py         preprocessing, tokenizer, ONNX sessions
  index.py             gallery index, z-score fusion, boosts, date filter
  query.py             free text → train no. / station / clean CLIP prompt
pipeline/              the ML pipeline (each file is one stage)
  fetch_seed_images.py demo photos from Wikimedia Commons (+ licence/author)
  ingest.py            raw → clean gallery + found_items.csv (privacy steps here)
  fetch_model.py       pinned model download + sha256 manifest
  build_index.py       embeds the gallery, writes index + lineage manifest
  evaluate.py          recall@k, MRR, precision@5, latency; --gate, --mlflow
app/main.py            API: /api/search, /api/feedback, /health, /metrics
app/static/index.html  the web UI
tests/                 23 tests that use tiny stand-in ONNX models (run in ~2 s)
params.yaml            every knob, versioned
dvc.yaml               pipeline DAG for `dvc repro`
Dockerfile             build stage (model + index + gate) → runtime stage
.github/workflows/ci.yml   tests → real-model quality gate → docker smoke test
```

---

## Run it today

Requires Python 3.11+ and git. Docker is optional for local runs. Vercel builds the image for you.

### 1. Set up (5 min)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                        # 23 passed
```

### 2. Build the demo gallery (20 min)
```bash
python -m pipeline.fetch_seed_images --per-category 8   # ~130 photos, 16 categories
```
Open `data/raw/commons/`. **Delete every photo that isn't clearly one lost-able item**, such as shop shelves, diagrams or crowds. This is real data cleaning. Keep at least 3 photos per category.

**Strongly recommended:** add 10–20 photos of your own things, like your backpack, bottle or umbrella, one folder per category:
```
data/raw/own/backpack/IMG_2041.jpg
data/raw/own/water_bottle/IMG_2050.jpg
```
These make the live demo convincing, because you can search for your own bag in front of an interviewer.

```bash
python -m pipeline.ingest     # → data/gallery/*.jpg + data/found_items.csv
```

### 3. Run the ML pipeline (5 min, mostly the ~150 MB download)
```bash
python -m pipeline.fetch_model
python -m pipeline.build_index
python -m pipeline.evaluate --gate
```
Read the evaluation table. If a gate fails, look at the "weakest categories" line first; usually the fix is curating bad photos. Only lower the gate in `params.yaml` if the data is fine.

### 4. Try it locally
```bash
uvicorn app.main:app --reload --port 8000     # http://localhost:8000
```

### 5. Push to GitHub
```bash
git init && git add . && git commit -m "Khoya-Paya: CLIP lost & found matcher"
git branch -M main
git remote add origin https://github.com/<you>/khoya-paya.git
git push -u origin main
```
Commit `data/gallery/` and `data/found_items.csv`; Vercel builds from them. `data/raw/` and `artifacts/` are git-ignored. The CI workflow starts on push; check the **Actions** tab.

### 6. Deploy to Vercel (10 min)
```bash
npm i -g vercel
vercel login
vercel link                                    # create a new project
vercel project update --framework container --yes
vercel deploy --prod
```
Alternatively, import the GitHub repo at vercel.com/new. Every push to `main` then deploys automatically.

`vercel.json` already sets `"framework": "container"`, so Vercel builds the `Dockerfile` instead of detecting FastAPI. The first build downloads the model and runs the quality gate, so it takes a few minutes. Open `https://<your-app>.vercel.app/health` and check that the model revision appears.

### Optional: run the production container locally
```bash
docker build -t khoya-paya . && docker run --rm -p 8080:80 khoya-paya
```

---

## The MLOps concepts, and where they live

| Concept | Where | Interview question it answers |
|---|---|---|
| **Reproducibility** | `params.yaml` pins the model commit; `model_manifest.json` stores sha256 of every file | "How do you know prod runs the model you tested?" |
| **Lineage** | `artifacts/index/manifest.json` links data hash and model hash; exposed at `/health` | "Which data and model produced this prediction?" |
| **Pipelines as DAGs** | `dvc.yaml`: `dvc repro` reruns only stages whose inputs changed | "How do you avoid re-running everything?" |
| **Offline evaluation** | `pipeline/evaluate.py`: recall@1/5, MRR, precision@5, p50/p95 latency | "How do you measure a retrieval system with no labels?" (Synthetic augmentations for image→image, categories for text→image, a small human set for realism.) |
| **Quality gates / CI-CD** | `--gate` in the Docker build **and** in GitHub Actions | "What stops a bad model reaching users?" |
| **Model optimisation** | int8 dynamic quantisation via ONNX | "How would you serve this on CPU cheaply?" Measure it with `variant: fp32` vs `quantized`. |
| **Containerisation** | Multi-stage Dockerfile; runtime image has no torch | "Why is your image small, and why does it cold-start fast?" |
| **Observability** | JSON logs per request; Prometheus `/metrics`: latency, top-match z-score, claimed rank | "How do you know it's working after launch?" |
| **Feedback loop** | "This is mine" clicks → `/api/feedback` → logged with rank | "Where does training data for v2 come from?" |
| **Responsible AI** | Face blur, GPS/EXIF stripping, hashed query logs, photo attribution, synthetic-data disclaimer | "What privacy risks does this system have?" |

### Experiments to run (great for your README and interviews)
1. **Quantised vs fp32:** set `model.variant: fp32` and run fetch → index → evaluate. Compare recall and latency. Was a 4× size cut worth the quality change?
2. **Boost tuning:** change `search.train_boost` and watch `human_recall_at_5`.
3. **MLflow:** `pip install mlflow`, then `python -m pipeline.evaluate --mlflow` and `mlflow ui` to compare runs side by side.
4. **Human eval set:** copy `data/eval_queries.example.csv` to `data/eval_queries.csv`. Write about 20 queries the way a worried passenger would, using real item IDs from `found_items.csv`. This is your most honest metric.

---

## Monitoring in production

- **Vercel → Logs:** each search writes a JSON line with mode, filters used, `top_z`, latency and result count. It contains a hash of the query, never the raw text.
- **`/metrics`:** if `khoya_top_match_zscore` trends down, the gallery is drifting away from what people search for (e.g. many new item types). If `khoya_claimed_rank` mostly shows rank 1–3, ranking is working.
- Metrics are per-instance because Vercel scales containers horizontally. For real dashboards, ship logs to a log drain such as Grafana or Datadog.

---

## Honest limitations

- **The data is a demo.** Photos come from Wikimedia Commons and the found-item metadata (train, station, date) is **synthetic**. The real integration target is the RPF's Mission Amanat listings.
- **The gallery is fixed at build time.** Container disks aren't durable, so adding found items means a redeploy. Production needs object storage plus a database (see roadmap).
- **CLIP is general-purpose.** It knows "backpack" but not that a "Wildcraft" bag looks a certain way. It also handles Indian-specific items like tiffin boxes less well.
- **Haar-cascade face blurring misses** profiles and small faces. Use a stronger detector in production.
- **English queries only** (see roadmap).

## Roadmap
- [ ] **Officer upload flow:** RPF staff photograph found items. Use Vercel Blob for images and Postgres + pgvector for metadata and embeddings, so new items are searchable instantly with no redeploy.
- [ ] **Fine-tune CLIP** with a contrastive loss on Indian lost-item photos, starting from the feedback-click pairs. Export to ONNX and gate on the same eval.
- [ ] **Multilingual queries** (Hindi, Marathi, Tamil): swap the text encoder for a multilingual CLIP.
- [ ] **OCR on tags and name labels**, with automatic masking of ID numbers.
- [ ] **Drift dashboard and alerting** on the top-z and claimed-rank metrics.
- [ ] **Shadow-deploy a new model** and compare it against the current one before switching.

## Credits
Gallery photos: Wikimedia Commons contributors. The licence and author of each photo are stored in `data/found_items.csv` and shown on every result card. Model: OpenAI CLIP ViT-B/32, ONNX export by Xenova (Hugging Face).
