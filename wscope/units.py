"""领域量：保留原始表示，派生层才做规范化。

规则：
- 导入的标量一律保存原始字符串与坐标基准（basis）；
- 能转换的给出 SI 值，无法转换的留在原位置并附诊断，绝不丢弃；
- 不允许为了画图把未转换值静默替换成默认值。
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any

# 各维度到 SI 基准单位的换算系数
TIME_TO_S = {
    "ps": 1e-12, "ns": 1e-9, "us": 1e-6, "µs": 1e-6,
    "ms": 1e-3, "s": 1.0, "sec": 1.0,
}
VOLT_TO_S = {"uv": 1e-6, "µv": 1e-6, "mv": 1e-3, "v": 1.0}
VDIV_TO_S = {"mv/div": 1e-3, "v/div": 1.0}
ATTEN = {"x": 1.0, "": 1.0, "ratio": 1.0, "times": 1.0}

SI_UNIT = {
    "time": "s", "volt": "V", "vdiv": "V/div",
    "attenuation": "dimensionless", "dimensionless": "",
}

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RE = re.compile(rf"^\s*({_NUM})\s*([^\s]*?)\s*$")


@dataclasses.dataclass
class DomainValue:
    raw: str
    value: float | None
    unit: str | None
    si_value: float | None
    si_unit: str | None
    basis: str
    diagnostic: str | None = None
    code: str | None = None

    @property
    def ok(self) -> bool:
        return self.diagnostic is None

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _as_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    return str(raw)


def parse_value(raw: Any, dimension: str, basis: str) -> DomainValue:
    """解析带单位的领域量。dimension: time/volt/vdiv/attenuation/dimensionless。"""
    text = _as_text(raw)
    match = _RE.match(text)
    if not match:
        return DomainValue(text, None, None, None, None, basis,
                           f"无法解析标量表示 {text!r}", "UNPARSEABLE_SCALAR")
    number_text, unit = match.groups()
    try:
        number = float(number_text)
    except ValueError:
        return DomainValue(text, None, unit, None, None, basis,
                           f"数值无法解析 {number_text!r}", "UNPARSEABLE_NUMBER")
    table = {
        "time": (TIME_TO_S, "s"),
        "volt": (VOLT_TO_S, "V"),
        "vdiv": (VDIV_TO_S, "V/div"),
        "attenuation": (ATTEN, "dimensionless"),
        "dimensionless": (ATTEN, ""),
    }.get(dimension)
    if table is None:
        return DomainValue(text, number, unit or None, number, unit or None, basis,
                           f"未知维度 {dimension!r}", "UNKNOWN_DIMENSION")
    factors, _ = table
    key = unit.lower()
    if key in ("div", "格", "graticule"):
        return DomainValue(text, number, unit, None, None, basis,
                           "单位为格(div)，未提供每格对应物理量基准，无法转换；保留原始表示",
                           "UNITS_DIV_WITHOUT_SCALE")
    if key not in factors:
        return DomainValue(text, number, unit or None, None, None, basis,
                           f"未知单位 {unit!r}，无法转换到 {SI_UNIT[dimension]}；保留原始表示",
                           "UNKNOWN_UNIT")
    return DomainValue(text, number, unit or None, number * factors[key],
                       SI_UNIT[dimension], basis)


def parse_time_with_div_scale(raw: Any, seconds_per_div: float, basis: str) -> DomainValue:
    """在明确提供每格秒数时，把 div 坐标转换成秒；否则返回带诊断的原值。"""
    dv = parse_value(raw, "time", basis)
    if dv.ok or dv.value is None:
        return dv
    if dv.code == "UNITS_DIV_WITHOUT_SCALE":
        return DomainValue(dv.raw, dv.value, dv.unit, dv.value * seconds_per_div,
                           "s", basis + f"（按 {seconds_per_div:g} s/div 换算）")
    return dv
