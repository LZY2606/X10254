"""Derived layer: normalization happens here only. The source layer keeps raw
representations. NumPy computes resampling, multi-peak cross-correlation,
trigger jitter and measurement-window statistics.

Rules (see RULE_VERSION):
  R-TIME-1  non-uniform time axes: dt is taken from physical timestamps, never
            from index differences.
  R-MASK-1  missing samples and saturated runs never enter interpolation.
  R-ALIGN-1 alignment keeps every significant cross-correlation peak as a
            candidate; an ambiguous match is not silently picked.
  R-DESKEW-1 per-channel deskew is subtracted before jitter comparison.
  R-BOUND-1 trigger marks on the record boundary exclude a segment from jitter.
  R-WIN-1  measurement windows are versioned; a revision opens a new node.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .quantities import parse_quantity, parse_time_list

RULE_VERSION = "rules-2026.09"

REF_T0 = 0.0
REF_DT = 1e-9          # 1 ns reference grid
REF_N = 201
REF_TRIGGER_ID = "S1"
LAG_MIN, LAG_MAX = -80e-9, 80e-9
PEAK_TIE_TOL = 0.005  # relative tolerance for equal-strength alignment peaks
XCORR_MIN = 0.3
SAT_RUN_MIN = 2


def ref_grid() -> np.ndarray:
    return REF_T0 + np.arange(REF_N) * REF_DT


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Source interpretation -> physical waveform (raw stays untouched upstream)
# ---------------------------------------------------------------------------

def interpret_segment(seg: dict, calib: dict) -> dict:
    """Turn a raw segment fact plus its channel calibration into physical
    samples on physical times. Returns diagnostics rather than dropping data."""
    diag: list[dict] = []
    raw = seg["raw"]
    tb = raw["timebase"]
    irregular = False
    if tb.get("kind") == "list":
        parsed = parse_time_list(tb["times"])
        times = np.array(parsed["seconds"], dtype=float)
        if not parsed["ok"]:
            diag.append({"code": "bad_timebase", "detail": parsed["error"]})
        dts = np.diff(times)
        # Rule R-TIME-1: spacing is judged from the physical axis.
        irregular = bool(dts.size and np.max(dts) - np.min(dts) > 1e-12)
    else:
        q0 = parse_quantity(tb["t0"], "time")
        qd = parse_quantity(tb["dt"], "time")
        if not (q0.ok and qd.ok):
            diag.append({"code": "bad_timebase",
                         "detail": "uniform parameters unparseable"})
            times = np.array([])
        else:
            times = q0.value + np.arange(tb["n"]) * qd.value
        if qd.ok and not math.isclose(
                qd.value, REF_DT, rel_tol=0, abs_tol=1e-15):
            irregular = True

    samples = np.array(
        [np.nan if s is None else s for s in raw["samples"]], dtype=float)
    present = ~np.isnan(samples)

    gain = parse_quantity(calib["vertical_gain"], "v_per_code")
    atten = parse_quantity(calib["probe_attenuation"], "attenuation")
    if not gain.ok or not atten.ok:
        diag.append({
            "code": "unconvertible_quantity",
            "detail": {
                "vertical_gain": gain.to_dict(),
                "probe_attenuation": atten.to_dict()},
        })
        usable = False
        volts = np.full_like(samples, np.nan)
    else:
        usable = True
        volts = samples * gain.value * atten.value

    full_scale = raw.get("adc_full_scale")
    sat_mask = np.zeros(samples.shape, dtype=bool)
    if usable and full_scale is not None and present.any():
        at_edge = present & (np.abs(samples) >= float(full_scale) - 1e-9)
        sat_mask = _runs(at_edge, SAT_RUN_MIN)
        if sat_mask.any():
            diag.append({"code": "saturated_samples",
                         "detail": {"count": int(sat_mask.sum())}})
    if (~present).any():
        diag.append({"code": "missing_samples",
                     "detail": {"count": int((~present).sum())}})
    if irregular:
        diag.append({"code": "timebase_nonuniform",
                     "detail": "physical timestamps used; no index math"})

    trig = parse_quantity(raw["trigger_mark"], "time")
    deskew = parse_quantity(calib.get("deskew", "0 ns"), "time")
    if not trig.ok:
        diag.append({"code": "bad_trigger", "detail": trig.error})
    boundary = False
    if trig.ok and times.size:
        if trig.value <= times[0] + 1e-12 or trig.value >= times[-1] - 1e-12:
            boundary = True
            diag.append({"code": "trigger_boundary",
                         "detail": {"trigger": trig.value,
                                    "tmax": float(times[-1])}})
    if not deskew.ok:
        diag.append({"code": "bad_deskew", "detail": deskew.error})

    return {
        "segment_id": seg["segment_id"],
        "channel_id": seg.get("channel_id", "CH1"),
        "times": times,
        "volts": volts,
        "valid": present & ~sat_mask,
        "usable": usable and not any(d["code"] == "unconvertible_quantity"
                                     for d in diag),
        "irregular": irregular,
        "saturated": bool(sat_mask.any()),
        "missing": bool((~present).any()),
        "boundary": boundary,
        "diag": diag,
        "trigger_s": trig.value if trig.ok else None,
        "trigger_raw": raw["trigger_mark"],
        "deskew_s": deskew.value if deskew.ok else None,
    }


def _runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    i = 0
    n = mask.size
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def resample(view: dict, grid: np.ndarray) -> np.ndarray:
    """Linear resample onto ``grid``. Rule R-MASK-1: only interpolate when
    both bracketing samples are present and unsaturated; target points in
    missing or saturated intervals stay NaN."""
    t, v, valid = view["times"], view["volts"], view["valid"]
    out = np.full(grid.shape, np.nan)
    if t.size == 0:
        return out
    order = np.argsort(t)
    t, v, valid = t[order], v[order], valid[order]
    idx = np.searchsorted(t, grid)
    for k, g in enumerate(grid):
        j = int(idx[k])
        if j == 0 or j >= t.size:
            continue
        t1, t2 = t[j - 1], t[j]
        if not (valid[j - 1] and valid[j]):
            continue
        frac = (g - t1) / (t2 - t1)
        if frac < 0 or frac > 1:
            continue
        out[k] = v[j - 1] + frac * (v[j] - v[j - 1])
    return out


def candidate_lags(wave: np.ndarray, ref: np.ndarray,
                   wave_valid: np.ndarray) -> list[dict]:
    """Normalized cross-correlation against the reference waveform, evaluated
    only where both sides are valid. Returns every significant peak so a
    multi-peak match stays visible (Rule R-ALIGN-1)."""
    grid = ref_grid()
    lag_steps = np.arange(int(LAG_MIN / REF_DT), int(LAG_MAX / REF_DT) + 1)
    scores: list[float] = []
    for step in lag_steps:
        shifted = np.roll(wave, int(step))
        mask = wave_valid & ~np.isnan(ref) & ~np.isnan(shifted)
        if step >= 0:
            mask[:step] = False
        else:
            mask[step:] = False
        a = shifted[mask]
        b = ref[mask]
        if a.size < 20 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            scores.append(-1.0)
            continue
        corr = float(np.mean((a - a.mean()) * (b - b.mean()))
                     / (a.std() * b.std()))
        scores.append(corr)
    scores_arr = np.array(scores)
    best = float(scores_arr.max())
    if best < XCORR_MIN:
        return []
    peaks = []
    for i, score in enumerate(scores_arr):
        if i in (0, len(scores_arr) - 1):
            continue
        if score < best - PEAK_TIE_TOL or score < XCORR_MIN:
            continue
        if score >= scores_arr[i - 1] and score >= scores_arr[i + 1]:
            lag_ns = float(lag_steps[i])
            peaks.append({"lag_ns": lag_ns, "score": round(float(score), 6)})
    # merge adjacent plateau bins, keep the bin nearest zero per plateau
    merged: list[dict] = []
    for p in sorted(peaks, key=lambda p: (abs(p["lag_ns"]), p["lag_ns"])):
        if merged and abs(p["lag_ns"] - merged[-1]["lag_ns"]) <= 1:
            continue
        merged.append(p)
    merged.sort(key=lambda p: (-p["score"], abs(p["lag_ns"])))
    return merged


def jitter_ns(lag_ns: float, view: dict, ref_trigger_s: float) -> float | None:
    """Jitter = matched pulse offset minus trigger offset minus deskew."""
    if view["trigger_s"] is None or view["deskew_s"] is None:
        return None
    return lag_ns - (view["trigger_s"] - ref_trigger_s) * 1e9 - \
        view["deskew_s"] * 1e9
