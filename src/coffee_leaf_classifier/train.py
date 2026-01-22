import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from google.cloud import storage
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from coffee_leaf_classifier.data import CoffeeLeafDataset
from coffee_leaf_classifier.model import Model


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a file to GCS."""
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    bucket_name, blob_path = gcs_path.split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    logger.info("Starting training...")

    batch_size = int(cfg.training.batch_size)
    num_workers = int(getattr(cfg.training, "num_workers", 0))
    logger.info(f"Using DataLoader num_workers={num_workers}")

    pin_memory = bool(getattr(cfg.training, "pin_memory", False))
    persistent_workers = bool(getattr(cfg.training, "persistent_workers", False)) and num_workers > 0
    prefetch_factor = int(getattr(cfg.training, "prefetch_factor", 2)) if num_workers > 0 else None

    train_ds = CoffeeLeafDataset("train")
    val_ds = CoffeeLeafDataset("validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model = Model(
        num_classes=int(cfg.model.params.num_classes),
        lr=float(cfg.training.learning_rate),
    )

    wandb_logger = None
    if cfg.experiment.logging.enable and cfg.experiment.logging.wandb.enable:
        wandb_logger = WandbLogger(
            project=cfg.experiment.logging.wandb.project,
            entity=cfg.experiment.logging.wandb.get("entity"),
            name=cfg.experiment.name,
            tags=list(cfg.experiment.logging.wandb.tags),
            offline=str(cfg.experiment.logging.wandb.mode).lower() == "offline",
            log_model=False,
        )
        wandb_logger.experiment.config.update(
            {
                "experiment": dict(cfg.experiment),
                "model": dict(cfg.model),
                "training": dict(cfg.training),
            },
            allow_val_change=True,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        logger=wandb_logger,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model saved at: {best_model_path}")

    gcs_model_path = os.environ.get("GCS_MODEL_PATH")
    if gcs_model_path and best_model_path:
        upload_to_gcs(best_model_path, gcs_model_path)

    logger.info("Training complete")


if __name__ == "__main__":
    train()
