"""Unit-aware quantity parsing.

The import layer keeps the *original representation* of every domain value.
Only the derivation layer asks for normalization; values that cannot be
converted stay in place and produce a diagnostic instead of being dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Coordinate bases used in the project. Every physical quantity carried by the
# derivation layer is expressed in one of these reference units.
BASE_UNITS = {
    "time": {"s", "ns", "us", "ms"},
    "voltage": {"V", "mV", "uV"},
    "frequency": {"Hz", "kHz", "MHz"},
    "gain": {"V/count", "mV/count"},
    "attenuation": {"x", "X"},
}

TIME_TO_S = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
VOLT_TO_V = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}
GAIN_TO_V_PER_COUNT = {"V/count": 1.0, "mV/count": 1e-3}
ATTEN_FACTOR = {"x": 1.0, "X": 1.0}


@dataclass(frozen=True)
class Quantity:
    """A value together with its original textual representation."""

    value: float
    unit: str
    raw: Any
    base: str | None

    def in_base(self) -> float:
        return self.value


class Diagnostic(Exception):
    """Raised when an original value cannot be converted to a known base."""

    def __init__(self, code: str, message: str, raw: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.raw = raw

    def as_event(self) -> dict:
        return {"kind": "diagnostic", "code": self.code,
                "message": self.message, "raw": self.raw}


def _split(raw: Any) -> tuple[float | None, str | None]:
    if isinstance(raw, bool):
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    if isinstance(raw, str):
        text = raw.strip()
        # split number suffix: "12 ns", "-3.5ns", "10x"
        i = 0
        if text[:1] in "+-":
            i = 1
        while i < len(text) and (text[i].isdigit() or text[i] in ".eE+-"):
            i += 1
        num_part = text[:i].strip()
        unit_part = text[i:].strip()
        try:
            return float(num_part), unit_part or None
        except ValueError:
            return None, None
    return None, None


def parse(raw: Any, accepted: dict[str, float], default_unit: str | None = None,
          code: str = "unconvertible-value") -> Quantity:
    """Parse ``raw`` against an accepted unit table.

    The original representation is always retained on ``raw``. A bare number
    uses ``default_unit``. An unknown/missing unit or non-numeric body raises
    :class:`Diagnostic`; callers leave the value at its source slot and record
    the diagnostic rather than dropping it.
    """
    value, unit = _split(raw)
    if value is None:
        raise Diagnostic(code, f"cannot interpret {raw!r} as a quantity", raw)
    if unit is None:
        unit = default_unit
    if unit not in accepted:
        raise Diagnostic(code, f"unknown unit {unit!r} for {raw!r}", raw)
    return Quantity(value, unit, raw, None)


def to_seconds(raw: Any, default_unit: str = "s") -> Quantity:
    q = parse(raw, TIME_TO_S, default_unit, "bad-time-unit")
    return Quantity(q.value * TIME_TO_S[q.unit], "s", q.raw, "time")


def to_volts(raw: Any, default_unit: str = "V") -> Quantity:
    q = parse(raw, VOLT_TO_V, default_unit, "bad-voltage-unit")
    return Quantity(q.value * VOLT_TO_V[q.unit], "V", q.raw, "voltage")


def gain_volts_per_count(raw: Any, default_unit: str = "V/count") -> Quantity:
    q = parse(raw, GAIN_TO_V_PER_COUNT, default_unit, "bad-gain-unit")
    return Quantity(q.value * GAIN_TO_V_PER_COUNT[q.unit], "V/count",
                    q.raw, "gain")


def attenuation_factor(raw: Any, default_factor: float = 1.0) -> Quantity:
    """Probe attenuation such as ``"10x"`` -> factor 10 (unitless base)."""
    if raw is None:
        return Quantity(default_factor, "x", raw, "attenuation")
    q = parse(raw, ATTEN_FACTOR, "x", "bad-probe-attenuation")
    return Quantity(q.value, "x", q.raw, "attenuation")


def format_base(seconds_or_value: float, base: str) -> str:
    if base == "time":
        return f"{seconds_or_value * 1e9:.4g} ns"
    if base == "voltage":
        return f"{seconds_or_value:.4g} V"
    return f"{seconds_or_value:.4g}"
