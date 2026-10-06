"""Smoke tests for cross-cutting utilities (Phase 0 bootstrap)."""

import logging
import random

from common.logging import configure_logging, stage_log
from common.seed import set_seed


def test_set_seed_is_deterministic():
    set_seed(42)
    first = [random.random() for _ in range(5)]
    set_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_seed_reports_available_backends():
    status = set_seed(42)
    assert status["random"] == "set"
    # numpy/tensorflow may or may not be installed during Phase 0.
    assert status["numpy"] in {"set", "unavailable"}
    assert status["tensorflow"] in {"set", "unavailable"}


def test_configure_logging_sets_root_level():
    configure_logging(level="WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_stage_log_emits_tagged_message(caplog):
    logger = logging.getLogger("gait.test.stage")
    with caplog.at_level(logging.INFO):
        stage_log(logger, "TRAIN", "Starting LSTM search")
    assert "[TRAIN] Starting LSTM search" in caplog.text
