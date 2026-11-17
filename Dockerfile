# syntax=docker/dockerfile:1
# Multi-stage build:
#   stage "build"   downloads the pinned model, builds the index and runs the
#                   QUALITY GATE. If retrieval quality drops below params.yaml
#                   gates, `docker build` fails and nothing is deployed.
#   stage "runtime" contains only what serving needs.

ARG PYTHON=python:3.12-slim

# ---------------------------------------------------------------- build stage
FROM ${PYTHON} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY params.yaml .
COPY khoya/ khoya/
COPY pipeline/ pipeline/
RUN python -m pipeline.fetch_model

# Data changes more often than the model, so copy it after the download
# to keep the model layer cached between builds.
COPY data/ data/
RUN python -m pipeline.build_index && python -m pipeline.evaluate --gate

# -------------------------------------------------------------- runtime stage
FROM ${PYTHON} AS runtime
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY params.yaml .
COPY khoya/ khoya/
COPY app/ app/
COPY data/gallery/ data/gallery/
COPY --from=build /app/artifacts/model/ artifacts/model/
COPY --from=build /app/artifacts/index/ artifacts/index/
COPY --from=build /app/artifacts/metrics.json artifacts/metrics.json

# Vercel sends traffic to port 80 unless a PORT env var is set in the project.
EXPOSE 80
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-80} --workers 1 --no-access-log"]
