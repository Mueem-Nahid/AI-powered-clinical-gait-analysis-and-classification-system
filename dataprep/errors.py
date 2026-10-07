"""Exception and report types for data validation."""

from __future__ import annotations

from dataclasses import dataclass, field


class DataValidationError(ValueError):
    """Raised when the dataset fails a critical, unrecoverable check."""


@dataclass
class ValidationReport:
    """Outcome of ``validate_schema``: status, issues, and a summary profile."""

    status: str = "PASS"  # PASS | PASS_WITH_WARNINGS | FAIL
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"
