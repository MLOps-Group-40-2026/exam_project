from loguru import logger

from coffee_leaf_classifier.data import MyDataset
from coffee_leaf_classifier.model import Model


def train():
    """Train the coffee leaf classifier model."""
    logger.info("Starting training...")
    dataset = MyDataset("data/raw")  # noqa
    model = Model()  # noqa
    # TODO: add rest of your training code here
    logger.info("Training complete")


if __name__ == "__main__":
    logger.info("Running training script")
    train()
