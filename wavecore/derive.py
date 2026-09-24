"""Numeric derivation rules for waveform overlay review.

R1  Non-uniform time axis: never subtract sample indices to obtain dt;
    always difference the real timestamps.
R2  Missing and saturated samples never enter interpolation.
R3  Alignment keeps multi-peak cross-correlation candidates instead of
    collapsing to a single lag.
"""
from __future__ import annotations

import numpy as np

FULL_SCALE = 255


def uniform_axis(dt_s, count, t0_s=0.0):
    return t0_s + np.arange(count, dtype=float) * float(dt_s)


def explicit_axis(timestamps_s):
    t = np.asarray(list(timestamps_s), dtype=float)
    flags = []
    if t.size > 2:
        d = np.diff(t)
        if not np.allclose(d, d[0], rtol=1e-3, atol=1e-15):
            flags.append("non_uniform_timebase")
    return t, flags


def axis_dt_stats(t):
    """R1: dt statistics from real timestamps, never from index math."""
    d = np.diff(np.asarray(t, dtype=float))
    return {
        "dt_min": float(d.min()),
        "dt_max": float(d.max()),
        "dt_mean": float(d.mean()),
        "uniform": bool(np.allclose(d, d[0], rtol=1e-3, atol=1e-15)),
    }


def codes_to_volts(codes, gain_v_per_count, attenuation, offset=128.0):
    c = np.asarray(codes, dtype=float)
    return (c - float(offset)) * float(gain_v_per_count) * float(attenuation)


def sample_mask(codes):
    c = np.asarray(codes, dtype=float)
    missing = np.isnan(c)
    saturated = (~missing) & ((c >= FULL_SCALE) | (c <= 0.0))
    allowed = ~(missing | saturated)
    return {"missing": missing, "saturated": saturated, "allowed": allowed}


def mask_intervals(mask):
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0]
    return [[int(g[0]), int(g[-1])] for g in np.split(idx, cuts + 1)]


def masked_interp(t, v, allowed, t_query):
    """R2: linear interpolation only where both bracketing samples are allowed."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    allowed = np.asarray(allowed, bool)
    tq = np.asarray(t_query, float)
    out = np.full(tq.shape, np.nan)
    if t.size < 2:
        return out
    idx = np.clip(np.searchsorted(t, tq) - 1, 0, t.size - 2)
    in_range = (tq >= t[0]) & (tq <= t[-1])
    ok = in_range & allowed[idx] & allowed[idx + 1]
    span = t[idx + 1] - t[idx]
    w = np.where(span > 0, (tq - t[idx]) / np.where(span > 0, span, 1.0), 0.0)
    out[ok] = v[idx][ok] + w[ok] * (v[idx + 1][ok] - v[idx][ok])
    return out


def xcorr_candidates(ref, seg, max_lag, top_k=5, min_score=0.6):
    """R3: all local-peak lag candidates above min_score, best first."""
    ref = np.asarray(ref, float)
    seg = np.asarray(seg, float)
    ref = ref - np.nanmean(ref)
    seg = seg - np.nanmean(seg)
    norm = float(np.sqrt(np.nansum(ref ** 2) * np.nansum(seg ** 2))) or 1.0
    scores = []
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            a, b = ref[-lag:], seg[:lag]
        elif lag > 0:
            a, b = ref[:-lag], seg[lag:]
        else:
            a, b = ref, seg
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 8:
            continue
        scores.append((lag, float(np.nansum(a[m] * b[m]) / norm)))
    peaks = []
    for i in range(1, len(scores) - 1):
        lag, s = scores[i]
        if s >= scores[i - 1][1] and s >= scores[i + 1][1] and s >= min_score:
            peaks.append((lag, s))
    peaks.sort(key=lambda p: -p[1])
    return [{"lag_samples": int(l), "score": round(s, 6)} for l, s in peaks[:top_k]]


def shift(arr, lag):
    arr = np.asarray(arr, float)
    out = np.full(arr.shape, np.nan)
    if lag >= 0:
        out[lag:] = arr[: arr.size - lag]
    else:
        out[:lag] = arr[-lag:]
    return out


def jitter_residuals_ps(trigger_times_s):
    t = np.asarray(list(trigger_times_s), float)
    return (t - t.mean()) * 1e12


def window_metric(t_rel, v, start_s, end_s, metric):
    t = np.asarray(t_rel, float)
    v = np.asarray(v, float)
    m = (t >= start_s) & (t <= end_s) & ~np.isnan(v)
    if m.sum() == 0:
        return None
    if metric == "mean":
        return float(np.mean(v[m]))
    if metric == "p2p":
        return float(np.max(v[m]) - np.min(v[m]))
    raise ValueError(f"unknown metric {metric!r}")


def boundary_candidates(trigger_offset_samples, count):
    """Trigger exactly on a capture boundary: keep every legal treatment."""
    if trigger_offset_samples <= 0:
        side = "post-trigger"
    elif trigger_offset_samples >= count - 1:
        side = "pre-trigger"
    else:
        return []
    return [
        {"id": "include-partial", "treatment": "include",
         "note": f"use partial window ({side} data only)"},
        {"id": "exclude-segment", "treatment": "exclude",
         "note": "exclude segment from windowed measurements"},
    ]
