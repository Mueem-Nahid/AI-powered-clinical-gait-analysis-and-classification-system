"""Schema + quality validation and report generation for the gait dataset."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from dataprep.errors import ValidationReport
from dataprep.schema import (
    CANONICAL_CHANNELS,
    JOINT_VALID,
    LEG_VALID,
    REQUIRED_COLUMNS,
    TIME_RANGE,
    channel_name,
)


def validate_schema(df: pd.DataFrame) -> ValidationReport:
    """Validate a typed gait DataFrame and produce a report + summary profile.

    This is a pure function: it never mutates ``df`` and performs no I/O. It expects
    the loader's output (numeric ``time``/``angle``, nullable-integer enums) but is
    defensive against nulls.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
        return ValidationReport(status="FAIL", errors=errors, warnings=warnings)

    # Critical: nulls in identity / label / enumeration columns.
    for col in ("subject", "condition", "replication", "leg", "joint"):
        n_null = int(df[col].isna().sum())
        if n_null:
            errors.append(f"Column '{col}' has {n_null} null value(s)")

    # Critical: non-finite measurements.
    for col in ("time", "angle"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric")
        elif not bool(np.isfinite(pd.to_numeric(df[col], errors="coerce").dropna()).all()):
            errors.append(f"Column '{col}' has non-finite value(s)")

    # Critical: invalid enumeration values.
    invalid_leg = sorted(set(df["leg"].dropna().unique()) - LEG_VALID)
    if invalid_leg:
        errors.append(f"Column 'leg' has invalid value(s): {invalid_leg}")
    invalid_joint = sorted(set(df["joint"].dropna().unique()) - JOINT_VALID)
    if invalid_joint:
        errors.append(f"Column 'joint' has invalid value(s): {invalid_joint}")

    # Warning: duplicate rows.
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        warnings.append(f"{n_duplicates} duplicate row(s) detected")

    # Warning: missing measurement values (recoverable via interpolation in Phase 3).
    for col in ("time", "angle"):
        n_missing = int(df[col].isna().sum())
        if n_missing:
            warnings.append(f"{n_missing} missing {col} value(s)")

    # Warning: time outside the expected normalized gait-cycle range.
    time = pd.to_numeric(df["time"], errors="coerce")
    t_min, t_max = float(time.min()), float(time.max())
    if t_min < TIME_RANGE[0] or t_max > TIME_RANGE[1]:
        warnings.append(
            f"time outside expected {TIME_RANGE[0]}-{TIME_RANGE[1]} range "
            f"(found {t_min}-{t_max})"
        )

    # Warning: missing leg/joint signal combinations.
    present = set(
        zip(df["leg"].dropna().astype(int), df["joint"].dropna().astype(int))
    )
    missing_signals = set(product(sorted(LEG_VALID), sorted(JOINT_VALID))) - present
    if missing_signals:
        names = [channel_name(l, j) for l, j in sorted(missing_signals)]
        warnings.append(f"Missing expected signal(s): {', '.join(names)}")

    # Warning: extra unexpected columns.
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        warnings.append(f"Unexpected extra column(s): {', '.join(extra)}")

    profile = _build_profile(df)

    if errors:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    return ValidationReport(status=status, errors=errors, warnings=warnings, profile=profile)


def _build_profile(df: pd.DataFrame) -> dict:
    """Summarize dataset shape and distribution (all values are native Python types)."""
    n_trials = df.groupby(["subject", "condition", "replication"], dropna=False).ngroups
    # Class distribution is reported at the *trial* level, not the row level.
    trials = df[["subject", "condition", "replication"]].drop_duplicates()
    class_distribution = (
        trials["condition"].dropna().astype(int).value_counts().sort_index().to_dict()
    )
    pairs = {
        (int(l), int(j))
        for l, j in zip(df["leg"].dropna(), df["joint"].dropna())
        if int(l) in LEG_VALID and int(j) in JOINT_VALID
    }
    signals_available = {
        ch: ch in {channel_name(l, j) for l, j in pairs} for ch in CANONICAL_CHANNELS
    }
    return {
        "rows": int(len(df)),
        "unique_subjects": int(df["subject"].nunique()),
        "unique_conditions": int(df["condition"].nunique()),
        "unique_legs": int(df["leg"].nunique()),
        "unique_joints": int(df["joint"].nunique()),
        "n_trials": int(n_trials),
        "time_min": float(pd.to_numeric(df["time"], errors="coerce").min()),
        "time_max": float(pd.to_numeric(df["time"], errors="coerce").max()),
        "n_time_points": int(df["time"].nunique()),
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "signals_available": signals_available,
    }


def write_report(report: ValidationReport, output_dir: str | Path = "output") -> None:
    """Persist the quality report (text) and profile (JSON) under ``output/``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = dict(report.profile)
    profile["status"] = report.status

    lines = [
        "DATA QUALITY REPORT",
        "-------------------",
        f"Rows: {profile.get('rows', 0)}",
        f"Subjects: {profile.get('unique_subjects', 0)}",
        f"Conditions: {profile.get('unique_conditions', 0)}",
        f"Trials: {profile.get('n_trials', 0)}",
        f"Missing values: {profile.get('n_missing', 0)}",
        f"Duplicates: {profile.get('n_duplicates', 0)}",
        "",
        f"Status: {report.status}",
    ]
    if report.errors:
        lines += ["", "Errors:", *[f"  - {e}" for e in report.errors]]
    if report.warnings:
        lines += ["", "Warnings:", *[f"  - {w}" for w in report.warnings]]

    (output_dir / "data_quality_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
