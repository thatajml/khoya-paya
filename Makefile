# Shortcuts. On Windows without `make`, run the python commands shown after each target.
PY ?= python

.PHONY: setup seed ingest model index eval pipeline serve test docker-build docker-run

setup:        ## create the dev environment
	$(PY) -m pip install -r requirements-dev.txt

seed:         ## download ~130 CC-licensed demo photos from Wikimedia Commons
	$(PY) -m pipeline.fetch_seed_images --per-category 8

ingest:       ## raw photos -> data/gallery + data/found_items.csv
	$(PY) -m pipeline.ingest

model:        ## download the pinned CLIP ONNX model (~150 MB)
	$(PY) -m pipeline.fetch_model

index:        ## embed the gallery
	$(PY) -m pipeline.build_index

eval:         ## quality report + gate
	$(PY) -m pipeline.evaluate --gate

pipeline: ingest model index eval

serve:        ## local server at http://localhost:8000
	$(PY) -m uvicorn app.main:app --reload --port 8000

test:
	$(PY) -m pytest -q

docker-build:
	docker build -t khoya-paya .

docker-run:   ## same container Vercel runs, on http://localhost:8080
	docker run --rm -p 8080:80 khoya-paya
