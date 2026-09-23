"""单位与坐标基准处理。

规则：导入层保留原始表示；派生层才做规范化。
无法转换的值返回 None 并产生诊断，绝不为了画图丢掉原值。
"""

TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
VOLT_UNITS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}
GAIN_UNITS = {"V/div": 1.0, "mV/div": 1e-3, "uV/div": 1e-6}
RATIO_UNITS = {"x": 1.0, ":1": 1.0, "": 1.0}

_DIMS = {"time": TIME_UNITS, "voltage": VOLT_UNITS, "gain": GAIN_UNITS, "ratio": RATIO_UNITS}


def to_si(quantity, dim):
    """quantity = {"value": v, "unit": u}; 返回 (si_value, diagnostic_or_None)。"""
    if quantity is None:
        return None, "missing-quantity"
    if not isinstance(quantity, dict) or "value" not in quantity:
        return None, f"malformed-quantity:{quantity!r}"
    unit = str(quantity.get("unit", ""))
    table = _DIMS.get(dim)
    if table is None:
        return None, f"unknown-dimension:{dim}"
    if unit not in table:
        return None, f"unconvertible-unit:{unit}:for:{dim}"
    try:
        return float(quantity["value"]) * table[unit], None
    except (TypeError, ValueError):
        return None, f"non-numeric-value:{quantity['value']!r}"
