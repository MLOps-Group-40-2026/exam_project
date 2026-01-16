from torch.utils.data import Dataset

from coffee_leaf_classifier.data import CoffeeLeafDataset


def test_coffee_leaf_dataset():
    """Test the CoffeeLeafDataset class."""
    dataset = CoffeeLeafDataset("train")
    assert isinstance(dataset, Dataset)
