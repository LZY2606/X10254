"""Units and coordinate-basis handling.

Raw representations are preserved at import; only the derived layer
normalizes to SI. Values that cannot be converted stay in place and
produce diagnostics instead of being dropped for plotting convenience.
"""
from __future__ import annotations

TIME_TO_S = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
VOLT_PER_COUNT = {"V/count": 1.0, "mV/count": 1e-3}
ATTENUATION = {"1:1": 1.0, "10:1": 10.0, "100:1": 100.0}


class ConversionError(ValueError):
    """Raised when a quantity cannot be converted to SI."""


def time_to_seconds(value, unit):
    if unit not in TIME_TO_S:
        raise ConversionError(f"unknown time unit {unit!r} for value {value!r}")
    try:
        return float(value) * TIME_TO_S[unit]
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"cannot convert time {value!r} [{unit}]") from exc


def gain_to_v_per_count(value, unit):
    if unit not in VOLT_PER_COUNT:
        raise ConversionError(f"unknown gain unit {value!r} for value {value!r}")
    try:
        return float(value) * VOLT_PER_COUNT[unit]
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"cannot convert gain {value!r} [{unit}]") from exc


def attenuation_factor(text):
    if isinstance(text, (int, float)):
        return float(text)
    if text not in ATTENUATION:
        raise ConversionError(f"unknown probe attenuation {text!r}")
    return ATTENUATION[text]


def try_convert(fn, *args):
    """Returns (si_value, diagnostic). Never raises."""
    try:
        return fn(*args), None
    except ConversionError as exc:
        return None, str(exc)
