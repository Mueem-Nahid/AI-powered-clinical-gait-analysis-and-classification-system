"""Tests for data loading and schema validation (Phase 1)."""

import pandas as pd
import pytest

from dataprep import DataValidationError, load_gait_csv, validate_schema


def _valid_df() -> pd.DataFrame:
    """A small valid long-format frame: 1 subject, 1 condition, 1 replication,
    2 legs x 3 joints x 2 time points -> all six signals present."""
    rows = []
    for leg in (1, 2):
        for joint in (1, 2, 3):
            for t in (0, 1):
                rows.append(
                    {
                        "subject": 1,
                        "condition": 1,
                        "replication": 1,
                        "leg": leg,
                        "joint": joint,
                        "time": t,
                        "angle": 10.0 + t,
                    }
                )
    return pd.DataFrame(rows)


def test_valid_dataset_passes():
    report = validate_schema(_valid_df())
    assert report.status == "PASS"
    assert report.errors == []
    assert report.warnings == []


def test_missing_column_fails():
    df = _valid_df().drop(columns=["condition"])
    report = validate_schema(df)
    assert report.status == "FAIL"
    assert any("condition" in e for e in report.errors)


def test_null_subject_fails():
    df = _valid_df()
    df.loc[0, "subject"] = pd.NA
    report = validate_schema(df)
    assert report.status == "FAIL"
    assert any("subject" in e for e in report.errors)


def test_invalid_leg_fails():
    df = _valid_df()
    df.loc[0, "leg"] = 3
    report = validate_schema(df)
    assert report.status == "FAIL"
    assert any("leg" in e for e in report.errors)


def test_duplicate_row_warns():
    df = pd.concat([_valid_df(), _valid_df().iloc[[0]]], ignore_index=True)
    report = validate_schema(df)
    assert report.status == "PASS_WITH_WARNINGS"
    assert any("duplicate" in w for w in report.warnings)


def test_missing_angle_warns_not_fails():
    df = _valid_df()
    df.loc[0, "angle"] = float("nan")
    report = validate_schema(df)
    assert report.status == "PASS_WITH_WARNINGS"
    assert any("angle" in w for w in report.warnings)


def test_missing_signal_warns():
    df = _valid_df()
    df = df[~((df["leg"] == 2) & (df["joint"] == 3))]  # drop right_ankle
    report = validate_schema(df)
    assert report.status == "PASS_WITH_WARNINGS"
    assert any("right_ankle" in w for w in report.warnings)


def test_loader_missing_file_raises():
    with pytest.raises(DataValidationError, match="not found"):
        load_gait_csv("does/not/exist.csv")


def test_loader_non_numeric_angle_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "subject,condition,replication,leg,joint,time,angle\n"
        "1,1,1,1,1,0,not_a_number\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="angle"):
        load_gait_csv(bad)


def test_loader_real_dataset():
    df = load_gait_csv("data/gait.csv")
    assert len(df) == 181800
    assert list(df.columns)[:7] == [
        "subject",
        "condition",
        "replication",
        "leg",
        "joint",
        "time",
        "angle",
    ]
    report = validate_schema(df)
    assert report.status == "PASS"
    assert report.profile["unique_subjects"] == 10
    assert report.profile["n_trials"] == 300
    assert report.profile["class_distribution"] == {"1": 100, "2": 100, "3": 100}
    assert all(report.profile["signals_available"].values())
