"""Seeding for reproducible training runs."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every random number generator a training run draws from.

    Covers Python's ``random`` (exploration, replay sampling), NumPy and
    PyTorch (network initialisation). SUMO is deterministic for a given route
    file, and route files are seeded separately (see ``Environment.reset``).

    On CPU with a fixed thread count this makes a training run repeatable bit
    for bit. On GPU some kernels are non-deterministic, so expect small drift.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global generator for third-party code
    torch.manual_seed(seed)
