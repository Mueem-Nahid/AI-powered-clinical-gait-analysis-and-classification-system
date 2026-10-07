"""Fixed data contract for the clinical gait dataset (spec section 2.1).

Only the column names and the leg/joint enumerations are hard-coded here — these are
the spec's fixed schema. Subject count, condition labels, class distribution, and
sequence length are always inferred from the data itself.
"""

from __future__ import annotations

REQUIRED_COLUMNS = [
    "subject",
    "condition",
    "replication",
    "leg",
    "joint",
    "time",
    "angle",
]

# Fixed enumerations (part of the spec contract, not inferred from data).
LEG_VALID = {1, 2}       # 1=Left, 2=Right
JOINT_VALID = {1, 2, 3}  # 1=Hip, 2=Knee, 3=Ankle

TIME_RANGE = (0.0, 100.0)  # normalized gait-cycle percentage

LEG_NAMES = {1: "left", 2: "right"}
JOINT_NAMES = {1: "hip", 2: "knee", 3: "ankle"}

# Numeric columns that the loader coerces to nullable integer / float dtypes.
INTEGER_COLUMNS = ["condition", "replication", "leg", "joint"]
FLOAT_COLUMNS = ["time", "angle"]


def channel_name(leg: int, joint: int) -> str:
    """Return the canonical channel name, e.g. ``left_hip``."""
    return f"{LEG_NAMES[leg]}_{JOINT_NAMES[joint]}"


# Canonical order: left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle.
CANONICAL_CHANNELS = [
    channel_name(leg, joint) for leg in sorted(LEG_VALID) for joint in sorted(JOINT_VALID)
]
