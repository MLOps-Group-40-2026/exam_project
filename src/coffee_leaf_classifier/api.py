"""FastAPI for coffee leaf disease classification."""
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="Coffee Leaf Disease Classifier",
    description="API for classifying coffee leaf diseases",
    version="0.1.0",
)

#TODO: root endpoint that redirects to docs

# Disease classes from the dataset
# not used for now but may be useful later
CLASSES = ["Healthy", "Miner", "Phoma", "Red Spider Mite", "Rust"] #noqa


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
    return HealthResponse(status="healthy")


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Get API information."""
    return InfoResponse(
        name="Coffee Leaf Disease Classifier",
        version="0.1.0",
        description="Classifies coffee leaf images into disease categories",
        model_loaded=False,  # TODO: Update when model is loaded
    )

# TODO: validate file type and size
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    """
    Predict disease from coffee leaf image.

    Args:
        file: Uploaded image file

    Returns:
        Prediction with confidence scores
    """
    # For now, return mock response
    # TODO: actual model prediction logic
    mock_probabilities = {
        "Healthy": 0.85,
        "Miner": 0.05,
        "Phoma": 0.03,
        "Red Spider Mite": 0.02,
        "Rust": 0.05,
    }

    return PredictionResponse(
        prediction="Healthy",
        confidence=0.85,
        probabilities=mock_probabilities,
        message=f"Mock prediction for file: {file.filename}",
    )
