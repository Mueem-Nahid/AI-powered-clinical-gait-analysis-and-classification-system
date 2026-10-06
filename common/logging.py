"""Structured logging with stage tags for the AutoML pipeline."""

from __future__ import annotations

import logging
import sys

# Stage tags emitted by every major pipeline stage (see spec section 22).
STAGES = frozenset(
    {
        "DATA",
        "EDA",
        "SPLIT",
        "PREPROCESS",
        "TUNING",
        "TRAIN",
        "EVALUATE",
        "SELECT",
        "EXPLAIN",
        "REPORT",
        "REGISTRY",
        "API",
    }
)

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO", fmt: str = DEFAULT_FORMAT) -> None:
    """Configure the root logger once at process start."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
        force=True,
    )


def stage_log(logger: logging.Logger, stage: str, message: str) -> None:
    """Emit a stage-tagged log line, e.g. ``[TRAIN] Starting LSTM search``."""
    tag = stage.upper()
    logger.info("[%s] %s", tag, message)
