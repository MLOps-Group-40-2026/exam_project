import os
import random

import wandb


def test_connection():
    """
    W&B smoke test.

    This test must not require interactive login (CI/local).
    - If WANDB_API_KEY is set: run online.
    - Else: run in offline mode.
    """

    # Force offline mode if no key is configured
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        project="coffee_leaf_diseases",
        entity="mlops-group-40",
        name="initial-connection-test",
        reinit=True,
    )

    assert run is not None

    acc = 0.1 + random.random() * 0.8
    run.log({"test_accuracy": acc})

    run.finish()
