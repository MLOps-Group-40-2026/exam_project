from pathlib import Path
import hydra
from omegaconf import DictConfig
from loguru import logger
from coffee_leaf_classifier.data import MyDataset
from coffee_leaf_classifier.model import Model

# Absolute path to exam_project/configs, computed from this file location.
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    logger.info("Starting training...")
    dataset = MyDataset(cfg.experiment.paths.data_dir)  # noqa: F841
    model = Model()  # noqa: F841
    logger.info("Training complete")
    print("Loaded config:")
    print(cfg)

if __name__ == "__main__":
    logger.info("Running training script")
    train()
