"""Deterministic seed configuration for reproducible runs."""

from __future__ import annotations

import random


def set_seed(seed: int = 42) -> dict[str, str]:
    """Set the random seed across every library that is importable.

    Returns a mapping of ``library -> status`` so callers can log which
    sources of nondeterminism were actually pinned (useful for the
    reproducibility section of ``run_metadata.json``).
    """
    random.seed(seed)
    status: dict[str, str] = {"random": "set"}

    try:
        import numpy as np

        np.random.seed(seed)
        status["numpy"] = "set"
    except ImportError:
        status["numpy"] = "unavailable"

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        status["tensorflow"] = "set"
    except ImportError:
        status["tensorflow"] = "unavailable"

    return status
