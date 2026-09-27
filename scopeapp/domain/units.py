"""Quantities, units and coordinate bases.

Every domain quantity carries its unit / coordinate basis.  Raw
representations are preserved on import; only derived layers normalise.
Values that cannot be converted stay in place with a diagnostic and are
never dropped just to make plotting possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TIME_FACTORS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9, "ps": 1e-12}
VOLT_FACTORS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6}


@dataclass(frozen=True)
class Quantity:
    """A scalar measurement preserving its original textual representation."""

    value: float
    unit: str
    raw: str | None = None

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "raw": self.raw}


@dataclass
class ConversionDiagnostic:
    object_id: str
    field: str
    code: str
    message: str
    raw: object = None

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "field": self.field,
            "code": self.code,
            "message": self.message,
            "raw": self.raw,
        }


@dataclass
class UnitConversion:
    """Result of a conversion; ``ok`` is False when conversion is impossible."""

    si_value: float | None
    si_unit: str
    ok: bool
    raw: object
    diagnostic: ConversionDiagnostic | None = None

    def to_dict(self) -> dict:
        out = {
            "si_value": self.si_value,
            "si_unit": self.si_unit,
            "ok": self.ok,
            "raw": self.raw,
        }
        if self.diagnostic:
            out["diagnostic"] = self.diagnostic.to_dict()
        return out


def convert(value: object, factors: dict[str, float], si_unit: str,
            object_id: str, field_name: str) -> UnitConversion:
    """Convert ``value`` to SI; non-numeric / unknown units keep a diagnostic."""
    if isinstance(value, bool):
        raw_num = None
    elif isinstance(value, (int, float)):
        raw_num = float(value)
    elif isinstance(value, str):
        text = value.strip()
        unit = ""
        num_text = text
        for candidate in sorted(factors, key=len, reverse=True):
            if text.endswith(candidate):
                unit = candidate
                num_text = text[: -len(candidate)].strip()
                break
        try:
            raw_num = float(num_text)
        except ValueError:
            raw_num = None
        if raw_num is not None and unit == "":
            return UnitConversion(
                None, si_unit, False, value,
                ConversionDiagnostic(object_id, field_name, "unknown_unit",
                                     f"unit missing or unknown in {text!r}", value),
            )
        value = unit or None
    else:
        raw_num = None
    if raw_num is None:
        return UnitConversion(
            None, si_unit, False, value,
            ConversionDiagnostic(object_id, field_name, "non_numeric",
                                 "value is not numeric and stays at its origin", value),
        )
    factor = factors.get(value if isinstance(value, str) else si_unit, 1.0)
    if isinstance(value, str) and value not in factors:
        return UnitConversion(
            None, si_unit, False, value,
            ConversionDiagnostic(object_id, field_name, "unknown_unit",
                                 f"unknown unit {value!r}", value),
        )
    return UnitConversion(raw_num * factor, si_unit, True, value)


def time_quantity(value: object, object_id: str, field_name: str) -> UnitConversion:
    return convert(value, TIME_FACTORS, "s", object_id, field_name)


def voltage_quantity(value: object, object_id: str, field_name: str) -> UnitConversion:
    return convert(value, VOLT_FACTORS, "V", object_id, field_name)


def parse_probe_attenuation(value: object, object_id: str,
                            diagnostics: list[ConversionDiagnostic]) -> float | None:
    """Parse e.g. ``"10x"`` / ``10`` into a dimensionless attenuation factor.

    Unparseable attenuation yields a diagnostic and ``None``; the affected
    channel is normalised as x1 (flagged) rather than the value being dropped.
    """
    if isinstance(value, bool):
        parsed = None
    elif isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip().lower().rstrip("x×").strip()
        try:
            parsed = float(text)
        except ValueError:
            parsed = None
    else:
        parsed = None
    if parsed is None or parsed <= 0:
        diagnostics.append(ConversionDiagnostic(
            object_id, "probe_attenuation", "bad_attenuation",
            "probe attenuation is not a positive number; x1 assumed for derived view",
            value,
        ))
        return None
    return parsed
