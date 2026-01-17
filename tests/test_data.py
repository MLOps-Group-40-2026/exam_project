from __future__ import annotations

from typing import Dict

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from coffee_leaf_classifier.data import CoffeeLeafDataset


class _FakeHFDataset:
    """Minimal HF-style dataset (supports __len__ and __getitem__)."""

    def __init__(self, n: int = 3, num_classes: int = 5) -> None:
        self._items = []
        for i in range(n):
            img = Image.new("RGB", (256, 256), color=(i * 30, i * 30, i * 30))
            self._items.append({"image": img, "label": i % num_classes})

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        return self._items[idx]


def _fake_load_dataset(_name: str) -> Dict[str, _FakeHFDataset]:
    return {
        "train": _FakeHFDataset(n=4),
        "validation": _FakeHFDataset(n=2),
        "test": _FakeHFDataset(n=3),
    }


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_dataset_init_and_len(monkeypatch, split: str):
    import coffee_leaf_classifier.data as data_module

    monkeypatch.setattr(data_module, "load_dataset", _fake_load_dataset)

    ds = CoffeeLeafDataset(split=split, target_size=128)
    assert isinstance(ds, Dataset)
    assert len(ds) > 0


def test_getitem_has_expected_keys_and_types(monkeypatch):
    import coffee_leaf_classifier.data as data_module

    monkeypatch.setattr(data_module, "load_dataset", _fake_load_dataset)

    target_size = 128
    ds = CoffeeLeafDataset(split="train", target_size=target_size)

    sample = ds[0]
    assert isinstance(sample, dict)
    assert set(sample.keys()) == {"image", "label"}

    image = sample["image"]
    label = sample["label"]

    assert isinstance(image, torch.Tensor)
    assert tuple(image.shape) == (3, target_size, target_size)
    assert image.dtype.is_floating_point
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0

    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert label.ndim == 0
    assert int(label.item()) >= 0


def test_target_size_changes_image_shape(monkeypatch):
    import coffee_leaf_classifier.data as data_module

    monkeypatch.setattr(data_module, "load_dataset", _fake_load_dataset)

    ds_64 = CoffeeLeafDataset(split="train", target_size=64)
    ds_224 = CoffeeLeafDataset(split="train", target_size=224)

    assert tuple(ds_64[0]["image"].shape) == (3, 64, 64)
    assert tuple(ds_224[0]["image"].shape) == (3, 224, 224)
