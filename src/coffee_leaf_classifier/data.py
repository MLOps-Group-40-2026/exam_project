from pathlib import Path

from datasets import load_dataset, load_from_disk
import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset

# Path to DVC-tracked dataset
DVC_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "coffee_leaf_diseases"


class CoffeeLeafDataset(Dataset):
    """Dataset for coffee leaf disease classification.

    Loads data from local DVC-tracked folder if available, otherwise falls back
    to HuggingFace Hub. This allows proper data versioning via DVC while
    maintaining compatibility with environments that don't have DVC set up.
    """

    def __init__(self, split: str = "train", target_size: int = 128) -> None:
        if DVC_DATA_PATH.exists():
            self.ds = load_from_disk(str(DVC_DATA_PATH))[split]
        else:
            self.ds = load_dataset("brainer-fp66/coffee-leaf-diseases")[split]

        self.transform = transforms.Compose(
            [
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        item = self.ds[index]
        image = self.transform(item["image"].convert("RGB"))  # tensor [3, S, S]
        label = torch.tensor(int(item["label"]), dtype=torch.long)  # tensor []
        return {"image": image, "label": label}
