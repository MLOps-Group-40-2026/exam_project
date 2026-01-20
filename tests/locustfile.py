"""Load testing for Coffee Leaf Disease Classifier API using Locust."""

from locust import HttpUser, task, between


class CoffeeAPIUser(HttpUser):
    """Simulated user for load testing the Coffee Leaf API."""

    wait_time = between(1, 3)
    host = "https://coffee-api-485178670977.europe-west1.run.app"

    @task(3)
    def health_check(self):
        """Test the health endpoint (most frequent)."""
        self.client.get("/health")

    @task(2)
    def get_info(self):
        """Test the info endpoint."""
        self.client.get("/info")

    @task(1)
    def predict(self):
        """Test the predict endpoint with a sample image."""
        # Create a simple test image (1x1 red pixel PNG)
        import io
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="green")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        self.client.post(
            "/predict",
            files={"file": ("test_leaf.png", img_bytes, "image/png")},
        )
