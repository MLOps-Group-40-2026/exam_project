from datasets import load_dataset
import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset


class CoffeeLeafDataset(Dataset):
    def __init__(self, split: str = "train", target_size: int = 128) -> None:
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
