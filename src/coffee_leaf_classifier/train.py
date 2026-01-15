from coffee_leaf_classifier.data import MyDataset
from coffee_leaf_classifier.model import Model


def train():
    dataset = MyDataset("data/raw")  # noqa
    model = Model()  # noqa
    # TODO: add rest of your training code here


if __name__ == "__main__":
    train()
