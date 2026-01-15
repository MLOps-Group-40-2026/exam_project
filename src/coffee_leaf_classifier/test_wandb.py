import wandb
import random


def test_connection():
    run = wandb.init(project="coffee_leaf_diseases", entity="mlops-group-40", name="initial-connection-test")

    run = wandb.init(
        project="coffee_leaf_diseases",
        entity="mlops-group-40",
        name="initial-connection-test"
    )

    acc = 0.1 + random.random() * 0.8

    run.log({"test_accuracy": acc})

    print(f"Successfully logged to: {run.url}")

    run.finish()


if __name__ == "__main__":
    test_connection()
