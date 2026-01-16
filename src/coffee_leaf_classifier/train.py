from pathlib import Path

import hydra
import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from coffee_leaf_classifier.data import CoffeeLeafDataset
from coffee_leaf_classifier.model import Model


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    logger.info("Starting training...")

    batch_size = int(cfg.training.batch_size)
    num_workers = int(getattr(cfg.training, "num_workers", 0))

    train_ds = CoffeeLeafDataset("train")
    val_ds = CoffeeLeafDataset("validation")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = Model(
        num_classes=int(cfg.model.params.num_classes),
        lr=float(cfg.training.learning_rate),
    )

    wandb_logger = None
    if cfg.experiment.logging.enable and cfg.experiment.logging.wandb.enable:
        wandb_logger = WandbLogger(
            project=cfg.experiment.logging.wandb.project,
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

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        logger=wandb_logger,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info("Training complete")


if __name__ == "__main__":
    train()
