"""Unit / coordinate-reference handling.

All domain quantities carry a unit or coordinate basis. The source layer keeps
the *original representation* untouched; the derived layer normalises to SI
seconds/volts. Values that cannot be converted stay in place and carry a
diagnostic instead of being dropped for plotting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

TIME_UNITS_S = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
}

VOLT_UNITS_V = {
    "V": 1.0,
    "v": 1.0,
    "mV": 1e-3,
    "uV": 1e-6,
    "µV": 1e-6,
}


@dataclass(frozen=True)
class ConvertResult:
    value: Optional[float]
    diagnostic: Optional[str]
    basis: str

    @property
    def ok(self) -> bool:
        return self.value is not None and math.isfinite(self.value)


def convert_seconds(rep: Any) -> ConvertResult:
    """Convert a {value, unit, basis} representation to seconds."""
    if not isinstance(rep, dict):
        return ConvertResult(None, "bad_representation: expected object", "t")
    unit = rep.get("unit")
    val = rep.get("value")
    basis = rep.get("basis", "t")
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return ConvertResult(None, "bad_representation: missing numeric value", basis)
    if unit not in TIME_UNITS_S:
        return ConvertResult(
            None,
            f"unconvertible_time_unit: {unit!r} (no SI mapping; original kept)",
            basis,
        )
    return ConvertResult(float(val) * TIME_UNITS_S[unit], None, basis)


def convert_volts(rep: Any) -> ConvertResult:
    if not isinstance(rep, dict):
        return ConvertResult(None, "bad_representation: expected object", "V")
    unit = rep.get("unit")
    val = rep.get("value")
    basis = rep.get("basis", "V")
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return ConvertResult(None, "bad_representation: missing numeric value", basis)
    if unit not in VOLT_UNITS_V:
        return ConvertResult(
            None,
            f"unconvertible_volt_unit: {unit!r} (no SI mapping; original kept)",
            basis,
        )
    return ConvertResult(float(val) * VOLT_UNITS_V[unit], None, basis)
