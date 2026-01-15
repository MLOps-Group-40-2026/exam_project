from coffee_leaf_classifier.model import Model
from coffee_leaf_classifier.data import CoffeeLeafDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

def train(batch_size = 64):
    model = Model()
    trainer = pl.Trainer(max_epochs=10)

    train_ds = CoffeeLeafDataset("train")
    val_ds = CoffeeLeafDataset("validation")
    test_ds = CoffeeLeafDataset("test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    batch = next(iter(train_loader))
    print(batch["image"].shape, batch["label"].shape)  # (8, 3, 256, 256) (8,)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    
    train()
