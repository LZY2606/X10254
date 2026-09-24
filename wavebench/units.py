"""Domain quantities.

Every domain value carries a unit or a coordinate basis. Import keeps the
original representation intact; only the derived layer normalises to SI/ADC.
Values that cannot be converted stay in place with a diagnostic instead of
being dropped to make the picture look clean.
"""
from __future__ import annotations

import re
from typing import Any

# Multipliers to the coherent SI unit (seconds, volts, hertz, dimensionless).
_PREFIX = {
    "": 1.0,
    "s": 1.0,
    "V": 1.0,
    "Hz": 1.0,
    "x": 1.0,
    "code": 1.0,
    "div": 1.0,
    "%": 0.01,
    "dB": None,  # non-linear, not silently converted
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}

# Accepted unit spellings -> (base coherent unit, kind)
_UNITS = {
    "s": ("s", "time"),
    "sec": ("s", "time"),
    "secs": ("s", "time"),
    "ms": ("s", "time"),
    "us": ("s", "time"),
    "µs": ("s", "time"),
    "ns": ("s", "time"),
    "V": ("V", "voltage"),
    "mV": ("V", "voltage"),
    "uV": ("V", "voltage"),
    "Hz": ("Hz", "frequency"),
    "kHz": ("Hz", "frequency"),
    "MHz": ("Hz", "frequency"),
    "x": ("x", "ratio"),
    "%": ("x", "ratio"),
    "code": ("code", "adc"),
    "codes": ("code", "adc"),
    "lsb": ("code", "adc"),
    "div": ("div", "scope"),
    "V/div": ("V/div", "gain_div"),
    "mV/div": ("V/div", "gain_div"),
    "V/code": ("V/code", "gain_code"),
    "mV/code": ("V/code", "gain_code"),
    "s/div": ("s/div", "time_div"),
}

# Exact multiplier per accepted spelling (no ambiguous prefix guessing).
_MULT = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "ms": 1e-3, "us": 1e-6,
    "µs": 1e-6, "ns": 1e-9,
    "V": 1.0, "mV": 1e-3, "uV": 1e-6,
    "Hz": 1.0, "kHz": 1e3, "MHz": 1e6,
    "x": 1.0, "%": 0.01,
    "code": 1.0, "codes": 1.0, "lsb": 1.0,
    "div": 1.0,
    "V/div": 1.0, "mV/div": 1e-3,
    "V/code": 1.0, "mV/code": 1e-3,
    "s/div": 1.0,
}

_TOKEN_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*([^\s]*)\s*$"
)


def raw_repr(value: Any) -> str:
    """Original textual representation, preserved verbatim."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _split(raw: str):
    match = _TOKEN_RE.match(raw)
    if not match:
        return None, None
    number = float(match.group(1))
    unit = match.group(2)
    return number, unit


def quantity(raw: Any, basis: str, expected_kind: str | None = None) -> dict[str, Any]:
    """Parse ``raw`` into a quantity while keeping the original representation.

    Returns a dict with keys raw / value / unit / basis / kind / diagnostic.
    ``value`` is None (and a diagnostic is attached) when conversion is
    impossible; the raw value never leaves its original position.
    """
    text = raw_repr(raw)
    out: dict[str, Any] = {
        "raw": text,
        "value": None,
        "unit": None,
        "basis": basis,
        "kind": expected_kind,
        "diagnostic": None,
    }
    if isinstance(raw, bool):
        out["diagnostic"] = "boolean is not a numeric quantity"
        return out
    if isinstance(raw, (int, float)):
        out["value"] = float(raw)
        out["unit"] = _default_unit(expected_kind)
        return out
    number, unit = _split(text)
    if number is None:
        out["diagnostic"] = f"unparseable quantity: {text!r}"
        return out
    spec = _UNITS.get(unit or "")
    if spec is None:
        out["unit"] = unit or None
        out["diagnostic"] = (
            f"unknown unit {unit!r}; value kept on basis {basis!r}, not normalised"
        )
        return out
    coherent_unit, kind = spec
    if expected_kind is not None and kind not in (expected_kind, "scope"):
        out["unit"] = coherent_unit
        out["diagnostic"] = (
            f"unit {unit!r} is {kind!r}, expected {expected_kind!r}; "
            "value kept unnormalised"
        )
        return out
    mult = _MULT.get(unit or "")
    if mult is None:
        out["unit"] = coherent_unit
        out["diagnostic"] = (
            f"unit {unit!r} has no linear SI conversion; value kept unnormalised"
        )
        return out
    value = number * mult
    out["value"] = value
    out["unit"] = coherent_unit
    out["kind"] = kind
    return out


def _default_unit(kind: str | None) -> str | None:
    return {
        "time": "s",
        "voltage": "V",
        "ratio": "x",
        "adc": "code",
        "frequency": "Hz",
    }.get(kind or "", None)


def as_value(q: dict[str, Any]) -> float | None:
    return q.get("value") if isinstance(q, dict) else None


def is_convertible(q: dict[str, Any]) -> bool:
    return isinstance(q, dict) and q.get("value") is not None and not q.get("diagnostic")
