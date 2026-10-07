"""Read and coerce the raw gait CSV into a typed DataFrame.

The loader is the fail-fast boundary for *type* problems: a non-numeric value in a
numeric column raises ``DataValidationError`` naming the column. Missing cells become
``NaN`` and are handled later by the validator (as warnings) and preprocessing
(interpolation), never silently repaired here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataprep.errors import DataValidationError
from dataprep.schema import FLOAT_COLUMNS, INTEGER_COLUMNS, REQUIRED_COLUMNS


def load_gait_csv(path: str | Path) -> pd.DataFrame:
    """Load the gait CSV, normalize column names, and coerce dtypes.

    Raises ``DataValidationError`` if the file is missing, a required column is absent,
    or a numeric column contains non-numeric / non-integer values.
    """
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {', '.join(missing)}")

    for col in INTEGER_COLUMNS:
        df[col] = _coerce_int(df[col], col)
    for col in FLOAT_COLUMNS:
        df[col] = _coerce_float(df[col], col)

    ordered = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df[ordered]


def _coerce_int(series: pd.Series, col: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Column '{col}' must be numeric: {exc}") from exc
    non_null = numeric.dropna()
    if not non_null.apply(lambda x: float(x).is_integer()).all():
        raise DataValidationError(f"Column '{col}' must contain whole numbers")
    return numeric.astype("Int64")


def _coerce_float(series: pd.Series, col: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Column '{col}' must be numeric: {exc}") from exc
