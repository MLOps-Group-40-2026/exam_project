"""FastAPI for coffee leaf disease classification."""

import io
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import v2 as transforms

from coffee_leaf_classifier.model import Model

# Disease classes from the dataset
CLASSES = ["Healthy", "Miner", "Phoma", "Red Spider Mite", "Rust"]

# Global model variable
model: Model | None = None
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing (same as training)
IMAGE_SIZE = 128
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


def download_model_from_gcs(gcs_path: str, local_path: str) -> bool:
    """Download model from GCS bucket."""
    try:
        from google.cloud import storage

        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path[5:]

        bucket_name, blob_path = gcs_path.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded model from gs://{bucket_name}/{blob_path} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model from GCS: {e}")
        return False


def extract_image_features(image: Image.Image) -> dict:
    """Extract simple features from image for drift detection."""
    img_array = np.array(image)
    return {
        "brightness": float(np.mean(img_array)),
        "contrast": float(np.std(img_array)),
        "r_mean": float(np.mean(img_array[:, :, 0])),
        "g_mean": float(np.mean(img_array[:, :, 1])),
        "b_mean": float(np.mean(img_array[:, :, 2])),
    }


def log_prediction_to_gcs(
    features: dict,
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
) -> None:
    """Log prediction data to GCS for drift detection (background task)."""
    try:
        from google.cloud import storage

        bucket_name = os.environ.get("GCS_BUCKET", "mlops-group-40-2026-dvc")
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        timestamp = datetime.utcnow()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "features": features,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
        }

        blob_name = f"predictions/{timestamp.strftime('%Y/%m/%d/%H%M%S_%f')}.json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(log_entry), content_type="application/json")
        logger.debug(f"Logged prediction to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logger.warning(f"Failed to log prediction to GCS: {e}")


def load_model() -> Model:
    """Load the trained model from checkpoint, GCS, or create new one."""
    model_path = os.environ.get("MODEL_PATH", "models/model.ckpt")
    local_path = "/tmp/model.ckpt"

    if model_path.startswith("gs://"):
        if not Path(local_path).exists():
            download_model_from_gcs(model_path, local_path)
        model_path = local_path

    if Path(model_path).exists():
        logger.info(f"Loading model from checkpoint: {model_path}")
        loaded_model = Model.load_from_checkpoint(model_path, map_location=device)
    else:
        logger.warning(f"No checkpoint found at {model_path}, using untrained model")
        loaded_model = Model(num_classes=len(CLASSES))

    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    logger.info("Starting Coffee Leaf Disease Classifier API")
    model = load_model()
    logger.info(f"Model loaded successfully on {device}")
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Coffee Leaf Disease Classifier",
    description="API for classifying coffee leaf diseases",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: str
    confidence: float
    probabilities: dict[str, float]
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str


class InfoResponse(BaseModel):
    """Response model for app info."""

    name: str
    version: str
    description: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    logger.debug("Health check requested")
    return HealthResponse(status="healthy")


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Get API information."""
    return InfoResponse(
        name="Coffee Leaf Disease Classifier",
        version="0.1.0",
        description="Classifies coffee leaf images into disease categories",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile, background_tasks: BackgroundTasks) -> PredictionResponse:
    """
    Predict disease from coffee leaf image.

    Args:
        file: Uploaded image file
        background_tasks: FastAPI background tasks for async logging

    Returns:
        Prediction with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Prediction requested for file: {file.filename}")

    # Read and preprocess image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Extract features for drift detection
    features = extract_image_features(image)

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze()

    # Get prediction
    pred_idx = probabilities.argmax().item()
    confidence = probabilities[pred_idx].item()
    prob_dict = {cls: probabilities[i].item() for i, cls in enumerate(CLASSES)}

    logger.info(f"Prediction: {CLASSES[pred_idx]} with confidence {confidence:.2%}")

    # Log prediction to GCS in background (doesn't slow down response)
    background_tasks.add_task(
        log_prediction_to_gcs,
        features=features,
        prediction=CLASSES[pred_idx],
        confidence=confidence,
        probabilities=prob_dict,
    )

    return PredictionResponse(
        prediction=CLASSES[pred_idx],
        confidence=confidence,
        probabilities=prob_dict,
        message=f"Predicted disease for file: {file.filename}",
    )
