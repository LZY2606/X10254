"""推导规则层。

原始层(raw)一律保留导入时的表示(单位、非等间隔时间戳、null 缺样);
只有本模块把量规范化到 ns / V 供计算使用。

三条硬规则(与反例档案中的机器可读事件对应):
  R1 nonuniform_no_index_subtraction : 时间轴不等间隔时不用索引相减,
     所有时间差由实际时间戳计算。
  R2 invalid_samples_excluded_from_interp : 缺样(null)与饱和(满量程)样本
     不进入任何插值; 跨越无效区间的测量被判为 excluded 并给出原因。
  R3 multipeak_candidates_kept : 对齐保留多峰互相关候选, 不取唯一 argmax,
     由人工裁定事件决定采纳哪一峰。
"""
from __future__ import annotations

import numpy as np

UNIT_TO_NS = {"ns": 1.0, "ps": 1e-3, "us": 1e3, "s": 1e9, "samples": None}
UNIT_TO_V = {"V": 1.0, "mV": 1e-3}


class Diagnostic(Exception):
    """无法转换/无法计算时抛出, 调用方把诊断留在原位置。"""


def qty_to_ns(qty) -> float:
    """{value, unit} -> ns。unit=samples 或未知单位: 抛 Diagnostic。"""
    unit = qty.get("unit")
    factor = UNIT_TO_NS.get(unit)
    if factor is None:
        raise Diagnostic(f"cannot convert unit {unit!r} to ns")
    return float(qty["value"]) * factor


def qty_to_v(qty) -> float:
    unit = qty.get("unit")
    factor = UNIT_TO_V.get(unit)
    if factor is None:
        raise Diagnostic(f"cannot convert unit {unit!r} to V")
    return float(qty["value"]) * factor


def normalize_timebase(timebase: dict, n_samples: int) -> np.ndarray:
    """时基规范化 -> 每样本时间戳(ns)。

    uniform: t0 + i*dt (仅此处允许用索引, 因为声明了等间隔)。
    nonuniform: 直接使用原始 timestamps, 绝不用 index*dt 近似 (R1)。
    """
    kind = timebase.get("kind")
    if kind == "uniform":
        dt = qty_to_ns(timebase["dt"])
        t0 = qty_to_ns(timebase["t0"])
        return t0 + dt * np.arange(n_samples, dtype=float)
    if kind == "nonuniform":
        ts = timebase.get("timestamps")
        if ts is None or len(ts) != n_samples:
            raise Diagnostic("nonuniform timebase timestamps length mismatch")
        return np.array([qty_to_ns(t) for t in ts], dtype=float)
    raise Diagnostic(f"unknown timebase kind {kind!r}")


def sample_values_and_mask(samples: list, full_scale_v: float, gain: float = 1.0):
    """返回 (values_v, valid_mask)。

    samples 元素为 {value, unit} 或 null(缺样)。
    饱和判定: |v| >= full_scale_v - 1e-12。缺样与饱和 -> mask False (R2)。
    gain 为探头衰减等外部增益, 原始样本先乘回真实电压。
    """
    vals = np.full(len(samples), np.nan)
    mask = np.zeros(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        if s is None:
            continue
        v = qty_to_v(s) * gain
        vals[i] = v
        if abs(v) < full_scale_v - 1e-12:
            mask[i] = True
    return vals, mask


def crossing_time(times: np.ndarray, vals: np.ndarray, mask: np.ndarray,
                  threshold: float) -> tuple[float | None, str | None]:
    """线性插值求上升沿过阈时间(ns)。

    只在使用两个有效样本的区间上插值 (R2); 若过阈区间任一端缺样/饱和,
    返回 (None, reason)。
    """
    for i in range(len(times) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not (mask[i] and mask[i + 1]):
            # 区间含无效样本, 但若信号明显跨越阈值, 需要报告被排除
            if not (np.isnan(v0) or np.isnan(v1)):
                if (v0 - threshold) * (v1 - threshold) < 0:
                    return None, f"crossing interval [{i},{i+1}] touches invalid sample"
            continue
        if (v0 - threshold) * (v1 - threshold) < 0 and v1 > v0:
            frac = (threshold - v0) / (v1 - v0)
            # R1: 用实际时间戳差, 不用 (i+1-i)*dt
            return float(times[i] + frac * (times[i + 1] - times[i])), None
    return None, "no rising crossing found in valid intervals"


def _corr_at_lag(seg_t, seg_v, seg_m, ref_t, ref_v, ref_m, lag):
    """把参考波形插值到 (seg_t - lag) 上求相关; 只用双方都有效的点 (R2)。"""
    t = seg_t - lag
    inside = (t >= ref_t[0]) & (t <= ref_t[-1]) & seg_m
    if inside.sum() < 8:
        return None
    # 参考侧插值位置若落在参考的无效区间, 该点也排除
    idx = np.searchsorted(ref_t, t[inside]).clip(1, len(ref_t) - 1)
    ok = ref_m[idx - 1] & ref_m[idx]
    if ok.sum() < 8:
        return None
    ti = t[inside][ok]
    vi = seg_v[inside][ok]
    ri = np.interp(ti, ref_t, ref_v)
    a = vi - vi.mean()
    b = ri - ri.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return None
    return float((a * b).sum() / denom)


def alignment_candidates(seg_t, seg_v, seg_m, ref_t, ref_v, ref_m,
                         lag_min=-2.0, lag_max=2.0, lag_step=0.01,
                         peak_ratio=0.6) -> list[dict]:
    """多峰互相关对齐候选 (R3)。

    在时延网格上评分, 返回所有 >= peak_ratio*max 的独立峰,
    按得分降序。时延以 ns 表示, 从不把样本索引差当时间差 (R1)。
    """
    lags = np.arange(lag_min, lag_max + lag_step / 2, lag_step)
    scores = np.full(len(lags), np.nan)
    for k, lag in enumerate(lags):
        c = _corr_at_lag(seg_t, seg_v, seg_m, ref_t, ref_v, ref_m, lag)
        if c is not None:
            scores[k] = c
    if np.all(np.isnan(scores)):
        return []
    best = np.nanmax(scores)
    thresh = best * peak_ratio
    cands = []
    k = 0
    n = len(lags)
    while k < n:
        if not np.isnan(scores[k]) and scores[k] >= thresh:
            j = k
            while j + 1 < n and not np.isnan(scores[j + 1]) and scores[j + 1] >= thresh:
                j += 1
            peak = k + int(np.nanargmax(scores[k:j + 1]))
            cands.append({"lag_ns": round(float(lags[peak]), 4),
                          "score": round(float(scores[peak]), 4)})
            k = j + 1
        else:
            k += 1
    cands.sort(key=lambda c: -c["score"])
    return cands


def histogram(values: list[float], bin_width: float) -> dict:
    """抖动分布直方图, bin 边界带单位语义(ns)。"""
    if not values:
        return {"bin_width_ns": bin_width, "bins": []}
    arr = np.array(values)
    lo = np.floor(arr.min() / bin_width) * bin_width
    hi = np.floor(arr.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + bin_width / 2, bin_width)
    counts, _ = np.histogram(arr, bins=edges)
    return {"bin_width_ns": bin_width,
            "bins": [{"lo_ns": round(float(edges[i]), 4),
                      "hi_ns": round(float(edges[i + 1]), 4),
                      "count": int(counts[i])} for i in range(len(counts))]}
