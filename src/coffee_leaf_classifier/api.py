"""FastAPI for coffee leaf disease classification."""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile
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


def load_model() -> Model:
    """Load the trained model from checkpoint or create new one."""
    model_path = os.environ.get("MODEL_PATH", "models/model.ckpt")

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
async def predict(file: UploadFile) -> PredictionResponse:
    """
    Predict disease from coffee leaf image.

    Args:
        file: Uploaded image file

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

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze()

    # Get prediction
    pred_idx = probabilities.argmax().item()
    confidence = probabilities[pred_idx].item()
    prob_dict = {cls: probabilities[i].item() for i, cls in enumerate(CLASSES)}

    logger.info(f"Prediction: {CLASSES[pred_idx]} with confidence {confidence:.2%}")

    return PredictionResponse(
        prediction=CLASSES[pred_idx],
        confidence=confidence,
        probabilities=prob_dict,
        message=f"Predicted disease for file: {file.filename}",
    )
