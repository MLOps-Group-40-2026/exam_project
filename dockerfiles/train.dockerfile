FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Use system Python (3.11 from PyTorch image) instead of downloading newer versions
ENV UV_PYTHON_PREFERENCE=only-system

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/

# Copy DVC config and data pointer files for runtime dvc pull
COPY .dvc .dvc/
COPY data/coffee_leaf_diseases.dvc data/

RUN uv sync --frozen

# Pull data from GCS at runtime (needs GCP credentials available on Vertex AI)
# Then run training
ENTRYPOINT ["sh", "-c", "uv run dvc pull data/coffee_leaf_diseases.dvc && uv run src/coffee_leaf_classifier/train.py"]
