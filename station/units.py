"""领域量解析：导入层只保留原始表示，派生层才做规范化。

无法转换的值留在原位置（raw_repr 不变），只附诊断，绝不为了画图而丢弃。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 时间：统一规范到纳秒 (ns)；电压增益规范到 V/ADC；衰减规范为无量纲比值。
_TIME_TO_NS = {
    "ns": 1.0,
    "us": 1e3,
    "µs": 1e3,
    "ms": 1e6,
    "s": 1e9,
}
_VOLT = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}


@dataclass
class Quantity:
    raw_repr: str
    unit: str
    coordinate_basis: str
    value: float | None = None          # 规范化后的数值；None 表示无法转换
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_repr": self.raw_repr,
            "unit": self.unit,
            "coordinate_basis": self.coordinate_basis,
            "value": self.value,
            "diagnostics": list(self.diagnostics),
        }


def parse_quantity(raw, unit, coordinate_basis, kind: str) -> Quantity:
    q = Quantity(raw_repr=raw, unit=unit, coordinate_basis=coordinate_basis)
    try:
        num = float(raw)
    except (TypeError, ValueError):
        q.diagnostics.append(
            f"{kind}: 原始表示 {raw!r} 无法转换为数值，保留原位不参与派生"
        )
        return q
    if kind == "time":
        factor = _TIME_TO_NS.get(unit)
        if factor is None:
            q.diagnostics.append(f"{kind}: 未知时间单位 {unit!r}")
            return q
        q.value = num * factor
    elif kind == "gain":
        # 原始单位 mV/ADC 或 V/ADC -> V/ADC
        factor = _VOLT.get(unit)
        if factor is None:
            q.diagnostics.append(f"{kind}: 未知电压单位 {unit!r}")
            return q
        q.value = num * factor
    elif kind == "attenuation":
        q.value = num  # 无量纲比值
    else:
        q.value = num
    return q


def parse_probe(raw) -> Quantity:
    """'10x' / '1x' / 数字。无法解析则留在原位。"""
    if isinstance(raw, (int, float)):
        return Quantity(str(raw), "ratio", "scope_probe", float(raw))
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*[xX×]\s*$", str(raw))
    if not m:
        q = Quantity(str(raw), "ratio", "scope_probe")
        q.diagnostics.append(f"probe: 探头衰减 {raw!r} 无法解析为 Nx，保留原位")
        return q
    return Quantity(str(raw), "ratio", "scope_probe", float(m.group(1)))
