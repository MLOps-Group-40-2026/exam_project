"""Tests for the FastAPI application.
These tests are for the mock API. Update when real model inference is added.
"""

import pytest
from fastapi.testclient import TestClient

from coffee_leaf_classifier.api import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


class TestInfoEndpoint:
    """Tests for /info endpoint."""

    def test_info_returns_200(self):
        """Info endpoint should return 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_contains_required_fields(self):
        """Info endpoint should contain all required fields."""
        response = client.get("/info")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "model_loaded" in data

    # should be changed when model is implemented
    def test_info_model_not_loaded(self):
        """Model should not be loaded in mock version."""
        response = client.get("/info")
        assert response.json()["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_returns_200(self, tmp_path):
        """Predict endpoint should return 200 OK with valid file."""
        # temporary test image file
        # TODO: replace with actual image content later
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image content")

        with open(test_image, "rb") as f:
            response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

        assert response.status_code == 200

    def test_predict_returns_required_fields(self, tmp_path):
        """Predict endpoint should return all required fields."""
        # temporary test image file
        # TODO: replace with actual image content later
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image content")

        with open(test_image, "rb") as f:
            response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "message" in data

    def test_predict_returns_valid_prediction(self, tmp_path):
        """Predict endpoint should return a valid disease class."""
        # temporary test image file
        # TODO: replace with actual image content later
        valid_classes = ["Healthy", "Miner", "Phoma", "Red Spider Mite", "Rust"]
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image content")

        with open(test_image, "rb") as f:
            response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

        data = response.json()
        assert data["prediction"] in valid_classes

    def test_predict_confidence_in_valid_range(self, tmp_path):
        """Confidence should be between 0 and 1."""
        # temporary test image file
        # TODO: replace with actual image content later
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image content")

        with open(test_image, "rb") as f:
            response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

        data = response.json()
        assert 0 <= data["confidence"] <= 1

    def test_predict_without_file_returns_422(self):
        """Predict endpoint should return 422 (unprocessable content)  when no file is provided."""
        response = client.post("/predict")
        assert response.status_code == 422
