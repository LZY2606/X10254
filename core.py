"""波形叠影台领域核心：归一化、插值、多峰对齐、抖动与测量窗口。

推导规则（机器可读，见 RULES，反例档案与事件负载引用这些规则 id）：
- nonuniform-timebase-no-index-subtraction:
  时间轴不等间隔时不用索引相减，时间差由实际时间轴逐点相减得到；
- gap-saturation-excluded-from-interpolation:
  缺样与饱和区间不进入插值；
- multi-peak-alignment-candidates-retained:
  对齐保留多峰互相关候选，无法唯一决定时保留全部合法候选。
"""
from __future__ import annotations

import numpy as np

RULES = {
    "nonuniform-timebase-no-index-subtraction":
        "时间轴不等间隔时不用索引相减，时间差由实际时间轴逐点相减得到",
    "gap-saturation-excluded-from-interpolation":
        "缺样与饱和区间不进入插值",
    "multi-peak-alignment-candidates-retained":
        "对齐保留多峰互相关候选，无法唯一决定时保留全部合法候选",
}

UNIT_TO_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}


class UnconvertibleUnit(Exception):
    """单位无法转换：值必须留在原位置并附诊断，不能为了画图丢掉。"""

    def __init__(self, unit):
        self.unit = unit
        super().__init__("unconvertible unit: %r" % (unit,))


def to_seconds(quantity):
    unit = quantity.get("unit")
    if unit not in UNIT_TO_SECONDS:
        raise UnconvertibleUnit(unit)
    return float(quantity["value"]) * UNIT_TO_SECONDS[unit]


def normalize_segment(seg, channel_deskew_s=0.0):
    """把一段原始捕获归一化为派生层使用的 t(s)/v(V) 与有效掩码。

    原始表示（adc 码、原始单位）不做不可逆清洗；无法转换的值保留在
    diagnostics 里并附原因。
    """
    diagnostics = []
    values = seg["samples"]["values"]
    n = len(values)
    codes = np.array([np.nan if v is None else float(v) for v in values], dtype=float)

    tb = seg["timebase"]
    if tb["kind"] == "uniform":
        t0 = to_seconds(tb["t0"])
        dt = to_seconds(tb["dt"])
        t = t0 + np.arange(n, dtype=float) * dt
        uniform = True
    else:
        unit = tb.get("unit", "s")
        raw_t = np.array([float(x) for x in tb["t"]], dtype=float)
        if unit not in UNIT_TO_SECONDS:
            diagnostics.append({
                "field": "timebase", "reason": "unconvertible-unit",
                "raw": {"unit": unit},
                "note": "无法转换的值留在原位置并附诊断",
            })
            t = raw_t
        else:
            t = raw_t * UNIT_TO_SECONDS[unit]
        uniform = False
    dt_nominal = float(np.median(np.diff(t))) if n > 1 else 0.0

    missing = set(int(i) for i in seg.get("missing", []))
    missing |= set(int(i) for i in np.nonzero(np.isnan(codes))[0])
    sat_code = seg.get("saturation_code")
    saturated = set()
    if sat_code is not None:
        saturated = set(int(i) for i in np.nonzero(codes >= float(sat_code))[0])

    valid = np.ones(n, dtype=bool)
    for i in missing | saturated:
        if 0 <= i < n:
            valid[i] = False
    valid &= ~np.isnan(codes)

    gain_q = seg["vertical_gain"]
    if gain_q.get("unit") != "V/adc_code":
        diagnostics.append({
            "field": "vertical_gain", "reason": "unconvertible-unit", "raw": gain_q,
            "note": "无法转换的值留在原位置并附诊断",
        })
        volts = codes.copy()
        v_unit = seg["samples"].get("unit", "adc_code")
    else:
        att_q = seg["probe_attenuation"]
        if att_q.get("unit") != "ratio":
            diagnostics.append({
                "field": "probe_attenuation", "reason": "unconvertible-unit",
                "raw": att_q, "note": "无法转换的值留在原位置并附诊断",
            })
            att = 1.0
        else:
            att = float(att_q["value"])
        volts = codes * float(gain_q["value"]) * att
        v_unit = "V"

    deskew_q = seg.get("deskew", {"value": 0.0, "unit": "s"})
    try:
        deskew_seg = to_seconds(deskew_q)
    except UnconvertibleUnit:
        diagnostics.append({
            "field": "deskew", "reason": "unconvertible-unit", "raw": deskew_q,
            "note": "无法转换的值留在原位置并附诊断，不参与派生计算",
        })
        deskew_seg = 0.0

    trigger_s = to_seconds(seg["trigger_time"])

    return {
        "segment_id": seg["segment_id"],
        "channel": seg["channel"],
        "t": t,
        "v": volts,
        "v_unit": v_unit,
        "valid": valid,
        "uniform": uniform,
        "dt_nominal": dt_nominal,
        "missing_idx": sorted(missing),
        "saturated_idx": sorted(saturated),
        "diagnostics": diagnostics,
        "deskew_s": float(channel_deskew_s) + float(deskew_seg),
        "trigger_s": trigger_s,
        "t0": float(t[0]),
        "t_end": float(t[-1]),
    }


def interp_values(t, v, valid, t_query):
    """仅用有效样点插值；缺样与饱和区间不进入插值。

    返回 (values, excluded)：落在缺样/饱和区间内部的查询点被标记为
    excluded 且值为 NaN，而不是跨区间插值出假数据。
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    tq = np.asarray(t_query, dtype=float)
    out = np.full(tq.shape, np.nan)
    excluded = np.ones(tq.shape, dtype=bool)
    idx = np.nonzero(valid)[0]
    if idx.size >= 2:
        out = np.interp(tq, t[idx], v[idx], left=np.nan, right=np.nan)
        excluded = np.isnan(out)
        for a, b in zip(idx[:-1], idx[1:]):
            if b > a + 1:
                lo, hi = t[a], t[b]
                gap = (tq > lo) & (tq < hi)
                excluded |= gap
    out[excluded] = np.nan
    return out, excluded


def index_lag_to_seconds(t, lag):
    """把相关峰的位置（样点 lag）换算成秒。

    规则 nonuniform-timebase-no-index-subtraction：时间差由实际时间轴
    逐点相减取平均，绝不用 lag * dt 这种索引相减的近似。
    """
    t = np.asarray(t, dtype=float)
    n = t.size
    length = abs(int(lag))
    if length == 0:
        return 0.0
    if length >= n:
        raise ValueError("lag out of range")
    span = float(np.mean(t[length:] - t[:-length]))
    return span if lag > 0 else -span


def alignment_candidates(ref, seg, max_candidates=3, threshold=0.4):
    """对 seg 相对 ref 做互相关，保留多峰候选（规则 3）。

    返回按得分降序的候选列表；调用方负责给候选稳定 id 并持久化。
    """
    grid_t = ref["t"][ref["valid"]]
    a = np.asarray(ref["v"][ref["valid"]], dtype=float)
    b, _ = interp_values(seg["t"], seg["v"], seg["valid"], grid_t)
    b = np.where(np.isnan(b), 0.0, b)
    a0 = a - a.mean()
    b0 = b - b.mean()
    norm = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    corr = np.correlate(b0, a0, mode="full")
    lags = np.arange(-(a0.size - 1), b0.size)
    scores = corr / norm if norm > 0 else np.zeros_like(corr)
    peaks = []
    for i in range(1, len(scores) - 1):
        if scores[i] >= scores[i - 1] and scores[i] > scores[i + 1] and scores[i] > 0:
            peaks.append(i)
    peaks.sort(key=lambda i: scores[i], reverse=True)
    if not peaks:
        return []
    best = scores[peaks[0]]
    candidates = []
    for rank, i in enumerate(peaks[:max_candidates], start=1):
        if best > 0 and scores[i] < threshold * best:
            break
        lag = int(lags[i])
        candidates.append({
            "rank": rank,
            "lag_index": lag,
            "lag_s": index_lag_to_seconds(grid_t, lag),
            "score": float(scores[i]),
        })
    return candidates


def measure_window(aligned_t, v, valid, start_s, end_s, n=25):
    """在测量窗内插值并统计；缺样/饱和区间落入窗口则整窗标记排除。"""
    tq = np.linspace(start_s, end_s, n)
    vals, excluded = interp_values(aligned_t, v, valid, tq)
    if np.all(np.isnan(vals)):
        return {"status": "excluded", "reasons": ["no-valid-samples-in-window"]}
    if excluded.any():
        return {"status": "excluded", "reasons": ["gap-or-saturation-in-window"]}
    return {
        "status": "ok",
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }
