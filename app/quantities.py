"""Domain quantities: every value keeps its raw representation plus a canonical
SI value. Values that cannot be converted stay in place with a diagnostic
instead of being dropped for plotting."""
from __future__ import annotations

import re
from dataclasses import dataclass

# SI prefix -> multiplier to the base unit (s, V, ...).
_PREFIX = {
    "": 1.0,
    "p": 1e-12, "P": 1e-12,
    "n": 1e-9, "N": 1e-9,
    "u": 1e-6, "U": 1e-6, "µ": 1e-6,
    "m": 1e-3,
    "k": 1e3, "K": 1e3,
    "M": 1e6,
}

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_MEAS_RE = re.compile(
    rf"^\s*(?P<num>{_NUM})\s*(?P<pre>pPnNuUµmkK M]?)(?P<core>s|V|S)\s*"
    r"(?:/\s*(?P<per>div|code|格))?\s*$"
)
_ATTEN_RE = re.compile(rf"^\s*(?P<num>{_NUM})\s*(?P<x>[x×])\s*$")

CANONICAL_UNIT = {"time": "s", "voltage": "V"}
KINDS = {
    "time": ("s", None),
    "t_per_div": ("s", "div"),
    "t_per_code": ("s", "code"),
    "voltage": ("V", None),
    "v_per_div": ("V", "div"),
    "v_per_code": ("V", "code"),
}


@dataclass(frozen=True)
class Quantity:
    raw: str
    kind: str
    value: float | None = None
    unit: str | None = None
    per: str | None = None
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "kind": self.kind,
            "value": self.value,
            "unit": self.unit,
            "per": self.per,
            "ok": self.ok,
            "error": self.error,
        }


def unparseable(raw: object, kind: str, error: str) -> Quantity:
    return Quantity(raw=raw if isinstance(raw, str) else repr(raw),
                    kind=kind, ok=False, error=error)


def parse_quantity(raw: object, kind: str) -> Quantity:
    """Parse a domain measurement. ``kind`` is one of KINDS or 'attenuation'.

    Retains the original string; ``value`` is in canonical base units
    (seconds, volts, or a dimensionless factor for attenuation)."""
    if not isinstance(raw, str):
        return unparseable(raw, kind, "not a string representation")
    if kind == "attenuation":
        m = _ATTEN_RE.match(raw)
        if not m:
            return unparseable(raw, kind, "expected attenuation like '10x'")
        return Quantity(raw=raw, kind=kind, value=float(m.group("num")),
                        unit="factor", ok=True)
    if kind not in KINDS:
        return unparseable(raw, kind, f"unknown kind {kind!r}")
    expect_unit, expect_per = KINDS[kind]
    m = _MEAS_RE.match(raw)
    if not m:
        return unparseable(raw, kind, f"expected {kind}, e.g. '1 ns'")
    core = m.group("core")
    base = {"s": "s", "S": "s", "V": "V"}[core]
    if base != expect_unit:
        return unparseable(raw, kind, f"unit {base!r} does not match kind {kind!r}")
    per = m.group("per")
    if expect_per and per != expect_per:
        return unparseable(raw, kind, f"expected /{expect_per} for {kind}")
    if not expect_per and per is not None:
        return unparseable(raw, kind, f"unexpected /{per} for {kind}")
    pre = m.group("pre").replace(" ", "")
    value = float(m.group("num")) * _PREFIX[pre]
    return Quantity(raw=raw, kind=kind, value=value, unit=base, per=per, ok=True)


def parse_time_list(raw_times: object) -> dict:
    """Parse a list of raw time strings, preserving originals and order."""
    if not isinstance(raw_times, list) or not raw_times:
        return {"ok": False, "error": "time list missing", "items": [],
                "seconds": []}
    items, seconds = [], []
    ok = True
    for i, raw in enumerate(raw_times):
        q = parse_quantity(raw, "time")
        items.append(q.to_dict())
        seconds.append(q.value)
        ok = ok and q.ok
    monotonic = all(
        a is not None and b is not None and b > a
        for a, b in zip(seconds, seconds[1:])
    )
    return {"ok": ok and monotonic, "monotonic": monotonic,
            "error": None if ok and monotonic else
                    ("non-monotonic or unparseable timestamps"),
            "items": items, "seconds": seconds}
