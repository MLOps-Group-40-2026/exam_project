import pytorch_lightning as pl
import torch
from torch import nn


class Model(pl.LightningModule):
    def __init__(self, num_classes: int = 5, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # This makes it independent of image_size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)  # [B, 128, 1, 1]
        x = torch.flatten(x, 1)  # [B, 128]
        x = self.dropout(x)
        return self.classifier(x)  # [B, num_classes]

    def training_step(self, batch, batch_idx):
        x = batch["image"]  # <-- FIX: dictionary access
        y = batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    model = Model(num_classes=5)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(out.shape)  # torch.Size([2, 5])
