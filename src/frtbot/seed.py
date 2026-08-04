"""Deterministic seeding for reproducible research runs.

Every stochastic component (NumPy sampling, scikit-learn estimators, PyTorch
model init/training) must be seeded from the single value passed here so that
walk-forward runs are byte-for-byte reproducible given identical inputs.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed all known RNG sources and return the seed for logging/fingerprinting."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
