"""Tests for the FastAPI application."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from coffee_leaf_classifier.api import app


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as c:
        yield c


def create_test_image(width: int = 128, height: int = 128, color: tuple = (0, 255, 0)) -> bytes:
    """Create a valid test image as bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


class TestInfoEndpoint:
    """Tests for /info endpoint."""

    def test_info_returns_200(self, client):
        """Info endpoint should return 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_contains_required_fields(self, client):
        """Info endpoint should contain all required fields."""
        response = client.get("/info")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "model_loaded" in data

    def test_info_model_loaded(self, client):
        """Model should be loaded."""
        response = client.get("/info")
        assert response.json()["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_returns_200(self, client):
        """Predict endpoint should return 200 OK with valid image."""
        image_bytes = create_test_image()
        response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})
        assert response.status_code == 200

    def test_predict_returns_required_fields(self, client):
        """Predict endpoint should return all required fields."""
        image_bytes = create_test_image()
        response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})

        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "message" in data

    def test_predict_returns_valid_prediction(self, client):
        """Predict endpoint should return a valid disease class."""
        valid_classes = ["Healthy", "Miner", "Phoma", "Red Spider Mite", "Rust"]
        image_bytes = create_test_image()
        response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})

        data = response.json()
        assert data["prediction"] in valid_classes

    def test_predict_confidence_in_valid_range(self, client):
        """Confidence should be between 0 and 1."""
        image_bytes = create_test_image()
        response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})

        data = response.json()
        assert 0 <= data["confidence"] <= 1

    def test_predict_probabilities_sum_to_one(self, client):
        """Probabilities should sum to approx 1."""
        image_bytes = create_test_image()
        response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})

        data = response.json()
        prob_sum = sum(data["probabilities"].values())
        assert 0.99 <= prob_sum <= 1.01

    def test_predict_invalid_file_returns_400(self, client):
        """Predict endpoint should return 400 for invalid image data."""
        response = client.post("/predict", files={"file": ("test.jpg", b"not an image", "image/jpeg")})
        assert response.status_code == 400

    def test_predict_without_file_returns_422(self, client):
        """Predict endpoint should return 422 when no file is provided."""
        response = client.post("/predict")
        assert response.status_code == 422
