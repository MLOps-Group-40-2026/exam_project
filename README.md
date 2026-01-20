# Coffee Leaf Disease Classification (MLOps course project)


## Overall goal
Build a reproducible machine learning system that classifies coffee leaf images into disease categories.
The goal of the project is to demonstrate machine learning operations practices (reproducible training, experiment tracking, application logging, profiling, testing, and a deployable inference mechanism).

By the end, we aim to have:
- a reproducible training pipeline (containerized with Docker) and tested by comparing model weights.
- tracked experiments (hyperparameters, metrics, and artifacts) with Weights & Biases.
- a saved model artifact and a prediction entry point (command line interface and a small API).
- basic automated tests and continuous integration.

## Framework
**Framework:** PyTorch Lightning

How it is integrated:
- A `LightningDataModule` to handle dataset download, preprocessing, transformations, and train/validation/test splits.
- A `LightningModule` to define the model, loss, metrics, optimizer, and learning-rate scheduler.
- A single training entry point will run the Lightning `Trainer`, log results to Weights & Biases, and write model artifacts to disk.
- Selected runs will include performance profiling using PyTorch Profiler.

## Data
**Dataset:** `brainer-fp66/coffee-leaf-diseases` (Hugging Face).

From the dataset viewer and dataset card:
- Total examples: 2,164
- Splits: train 1,511; validation 323; test 330
- Format: Parquet
- Download size: approximately 1.63 GB
- Classes (5): Healthy, Miner, Phoma, Red Spider Mite, Rust

## Models we expect to use
1. Baseline: ResNet-18 or ResNet-34
2. Stronger baseline: EfficientNet
3. Experiment with: Vision Transformer baseline (if time allows)

## Evaluation metrics
Primary metrics:
- Accuracy
- Macro F1-score (more robust when classes are imbalanced)

Secondary metrics:
- Confusion matrix
- Per-class precision and recall

## Tooling and stack choices
- Package manager: `uv` for dependency management and a committed lockfile
- Framework: PyTorch Lightning
- Version control: GitHub
- Application logging: Loguru
- Experiment tracking: Weights & Biases
- Experiment configuration: Hydra
- Profiling: PyTorch Profiler
- Reproducibility: Docker

## Reproducibility plan
- Commit `pyproject.toml` and the `uv` lockfile so dependency resolution is reproducible.
- Set random seeds in training entry points.
- Use Docker to standardize the runtime environment across machines.
- Record training configuration (hyperparameters, data transformations, model name, random seed) in Weights & Biases and version control using Hydra.

## Running the Project

### Quick Start (Local, No GCP Required)

```bash
# Install dependencies
uv sync

# Train locally (downloads data automatically)
uv run python src/coffee_leaf_classifier/train.py training.epochs=5

# Run API locally
uv run uvicorn src.coffee_leaf_classifier.api:app --host 0.0.0.0 --port 8000

# Run tests
uv run pytest tests/
```

### Docker (Local, no GCP Required)

```bash
# Build and run training
docker build -f dockerfiles/train.dockerfile -t coffee-train .
docker run coffee-train

# Build and run API
docker build -f dockerfiles/api.dockerfile -t coffee-api .
docker run -p 8000:8000 coffee-api
```

### Live Deployed API (No Setup Required)

The API is deployed on Google Cloud Run and publicly accessible at:

- **Docs:** https://coffee-api-485178670977.europe-west1.run.app/docs
- **Health:** https://coffee-api-485178670977.europe-west1.run.app/health
- **Predict:** `POST https://coffee-api-485178670977.europe-west1.run.app/predict` (upload image)

### GCP Infrastructure (Team Access Only)

The following GCP services are used (requires project access):
- **Vertex AI:** GPU training jobs
- **Cloud Run:** API hosting
- **Artifact Registry:** Docker images
- **Cloud Storage:** Model checkpoints and DVC data
