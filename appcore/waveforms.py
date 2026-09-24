"""Deterministic derivation rules for segmented oscilloscope captures.

Rule judgements are emitted as machine-readable events with stable codes:

* ``time-uneven``          -- per-point time differences must be used, never
  index subtraction;
* ``gap-skipped`` / ``saturation-skipped`` -- missing samples and saturated
  regions never enter interpolation;
* ``multipeak-candidates`` -- cross-correlation keeps every legal alignment
  candidate when a unique solution does not exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RULE_VERSION = "rules-2026.09.1"

GAP = None  # original representation of a missing sample
SAT_HIGH = 32767
SAT_LOW = -32768

# A segment is measurable only if at least this fraction of window points are
# present *and* unsaturated.
MIN_VALID_FRACTION = 0.5
# A correlation peak counts as a candidate when within this fraction of the
# global maximum.
PEAK_RELATIVE = 0.85
PEAK_PROMINENCE = 0.10


@dataclass
class TimeAxis:
    times: np.ndarray                 # seconds, relative to capture start
    uneven: bool
    diagnostics: list[dict] = field(default_factory=list)


def materialize_time(timebase: dict, n: int) -> TimeAxis:
    """Turn an original timebase representation into absolute seconds.

    Uniform representation: ``{"unit": "ns", "delta": 1.0, "origin": 0.0}``.
    Uneven representation: ``{"unit": "ns", "t": [..explicit per-sample..]}``.

    The uneven branch never does ``index * delta``; it consumes the explicit
    coordinates and emits the ``time-uneven`` judgement event.
    """
    unit = timebase.get("unit", "s")
    from appcore.units import TIME_TO_S
    scale = TIME_TO_S.get(unit)
    diagnostics: list[dict] = []
    if scale is None:
        diagnostics.append({"kind": "diagnostic", "code": "bad-time-unit",
                            "message": f"unknown timebase unit {unit!r}",
                            "raw": timebase})
        return TimeAxis(np.full(n, np.nan), False, diagnostics)
    if "t" in timebase:
        raw_t = np.asarray(timebase["t"], dtype=float) * scale
        if raw_t.shape[0] != n:
            diagnostics.append({"kind": "diagnostic",
                                "code": "timebase-length-mismatch",
                                "message": "explicit t length != samples",
                                "raw": {"t_len": raw_t.shape[0], "n": n}})
        diffs = np.diff(raw_t)
        uneven = bool(np.any(np.abs(diffs - diffs[0]) > 1e-12 * max(1.0, np.abs(diffs[0]))))
        if uneven:
            diagnostics.append({
                "kind": "judgement", "code": "time-uneven",
                "rule": RULE_VERSION,
                "message": "non-uniform time axis: per-point differences used; "
                           "index subtraction forbidden",
                "median_dt_s": float(np.median(diffs)),
                "max_dt_s": float(np.max(diffs)),
                "min_dt_s": float(np.min(diffs)),
            })
        return TimeAxis(raw_t, uneven, diagnostics)
    delta = float(timebase.get("delta", 1.0)) * scale
    origin = float(timebase.get("origin", 0.0)) * scale
    return TimeAxis(origin + np.arange(n) * delta, False, diagnostics)


def naive_index_seconds(timebase: dict, n: int) -> np.ndarray:
    """The *wrong* time axis produced by index subtraction on an uneven base.

    Used by tests to prove the rule matters: for an uneven timebase this
    differs materially from :func:`materialize_time`.
    """
    unit = timebase.get("unit", "s")
    from appcore.units import TIME_TO_S
    scale = TIME_TO_S[unit]
    if "delta" not in timebase:
        # callers only use this when a nominal delta exists for comparison
        dt = float(np.median(np.diff(np.asarray(timebase["t"])))) * scale
    else:
        dt = float(timebase["delta"]) * scale
    origin = float(timebase.get("origin", 0.0)) * scale
    return origin + np.arange(n) * dt


def valid_mask(samples: list) -> np.ndarray:
    """Boolean mask of present, unsaturated samples.

    Gaps (``None``/NaN) and rails (saturation codes) are excluded. They stay in
    the original sample array; only the mask prevents interpolation over them.
    """
    mask = np.zeros(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        if s is None:
            continue
        try:
            v = float(s)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        if v >= SAT_HIGH or v <= SAT_LOW:
            continue
        mask[i] = True
    return mask


def mask_regions(samples: list, mask: np.ndarray) -> list[dict]:
    """Describe gap / saturation regions (for the exclusion UI and tests)."""
    regions: list[dict] = []
    n = len(samples)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        start = i
        s = samples[i]
        if s is None or (isinstance(s, float) and not np.isfinite(s)):
            code = "gap"
        else:
            v = float(s)
            code = "saturation-high" if v >= SAT_HIGH else "saturation-low"
        while i < n and not mask[i] and _region_code(samples[i]) == code:
            i += 1
        regions.append({"code": code, "start_index": start,
                        "end_index": i, "length": i - start})
    return regions


def _region_code(sample) -> str:
    if sample is None or (isinstance(sample, float) and not np.isfinite(sample)):
        return "gap"
    v = float(sample)
    return "saturation-high" if v >= SAT_HIGH else "saturation-low"


def to_voltage(samples: list, vgain_v_per_count: float | None,
               probe_factor: float | None,
               mask: np.ndarray) -> np.ndarray:
    """Normalize valid raw counts to volts at the probe tip.

    When gain or probe attenuation is unconvertible (``None``) the voltage
    layer is unavailable; the raw counts remain visible and the segment gets a
    diagnostic rather than being silently drawn in fake volts.
    """
    out = np.full(len(samples), np.nan)
    if vgain_v_per_count is None or probe_factor is None:
        return out
    arr = np.asarray([np.nan if s is None else float(s) for s in samples])
    out[mask] = arr[mask] * vgain_v_per_count * probe_factor
    return out


def _resample_on_grid(times: np.ndarray, values: np.ndarray,
                      mask: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation that refuses to bridge gap/saturation regions.

    A grid point receives a value only when both bracketing source samples are
    valid; otherwise it is NaN (``gap-skipped`` / ``saturation-skipped``).
    """
    out = np.full(grid.shape, np.nan)
    order = np.argsort(times)
    ts, vs, ms = times[order], values[order], mask[order]
    idx = np.searchsorted(ts, grid) - 1
    for k, j in enumerate(idx):
        if j < 0 or j + 1 >= len(ts):
            continue
        if not ms[j] or not ms[j + 1]:
            continue
        x0, x1 = ts[j], ts[j + 1]
        if x1 == x0:
            continue
        out[k] = vs[j] + (vs[j + 1] - vs[j]) * (grid[k] - x0) / (x1 - x0)
    return out


@dataclass
class Candidate:
    lag_s: float
    score: float
    rank: int
    cid: str


@dataclass
class AlignmentResult:
    candidates: list[Candidate]
    ambiguous: bool
    diagnostics: list[dict]
    aligned_times: dict[str, np.ndarray]   # channel id -> shifted times (s)
    grid: np.ndarray


def align_segment(segment: dict, channels: dict, grid: np.ndarray | None = None,
                  candidate_seq: int = 0) -> AlignmentResult:
    """Cross-correlate the segment against its reference pulse, multi-peak.

    ``channels`` maps channel id -> {"time", "voltage", "mask", "deskew_s"}.
    Candidate lags are real seconds (works on uneven axes). Every peak above
    the relative threshold is retained as a legal candidate.
    """
    diagnostics: list[dict] = []
    # Reference channel is the first listed channel.
    ref_id = segment["reference_channel"]
    ref = channels[ref_id]
    ref_t = ref["time"] + float(ref.get("deskew_s") or 0.0)
    ref_v = ref["voltage"]
    ref_m = ref["mask"]

    if grid is None:
        # Common uniform analysis grid at 1/4 of the median source interval, so
        # sub-interval event offsets on uneven axes survive (index subtraction
        # is never used to place time). The grid starts before the earliest
        # template support so the full pulse contributes to every lag.
        pulse_min = float(np.min(np.asarray(segment["pulse"]["t"])))
        valid_t = ref_t[ref_m]
        if valid_t.size < 4:
            diagnostics.append({"kind": "judgement",
                                "code": "alignment-insufficient-valid",
                                "rule": RULE_VERSION,
                                "message": "not enough valid reference samples"})
            return AlignmentResult([], True, diagnostics, {}, np.array([]))
        dt_grid = float(np.median(np.diff(np.sort(valid_t)))) / 4.0
        start = min(valid_t[0], pulse_min)
        grid = np.arange(start, valid_t[-1], dt_grid)

    ref_grid = _resample_on_grid(ref_t, ref_v, ref_m, grid)
    gmask = np.isfinite(ref_grid)

    # Synthetic reference pulse is supplied by the segment case metadata; in
    # general the reference is segment 0's waveform. Here alignment compares
    # each segment to the canonical pulse shape passed via segment["pulse"].
    pulse = np.asarray(segment["pulse"]["samples"], dtype=float)
    pulse_t = np.asarray(segment["pulse"]["t"], dtype=float)
    p_valid = np.isfinite(pulse)
    # Correlate on the grid using FFT-free normalized NCC over shifts.
    lags, scores = _normalized_cross_correlation(
        grid, ref_grid, gmask, pulse_t, pulse, p_valid)

    peaks = _local_peaks(scores)
    if not peaks:
        diagnostics.append({"kind": "judgement", "code": "no-alignment-peak",
                            "rule": RULE_VERSION,
                            "message": "cross-correlation found no valid peak "
                                       "(template support met gap/rail)"})
        return AlignmentResult([], True, diagnostics, {}, grid)
    top = scores[peaks].max()
    accepted = [p for p in peaks
                if scores[p] >= top * PEAK_RELATIVE and
                scores[p] >= PEAK_PROMINENCE]
    accepted.sort(key=lambda p: (-scores[p], lags[p]))
    candidates = [
        Candidate(lag_s=float(lags[p]), score=float(scores[p]), rank=i + 1,
                  cid=f"cand-{segment['seg_id']}-{i + 1}")
        for i, p in enumerate(accepted)
    ]
    ambiguous = len(candidates) > 1
    diagnostics.append({
        "kind": "judgement",
        "code": "multipeak-candidates" if ambiguous else "single-peak",
        "rule": RULE_VERSION,
        "message": ("multiple legal cross-correlation peaks retained"
                    if ambiguous else "unique alignment peak"),
        "candidate_count": len(candidates),
    })
    aligned_times = {
        cid: ch["time"] + float(ch.get("deskew_s") or 0.0) - candidates[0].lag_s
        for cid, ch in channels.items()
    }
    return AlignmentResult(candidates, ambiguous, diagnostics,
                           aligned_times, grid)


def _normalized_cross_correlation(grid, sig, smask, pulse_t, pulse, pmask):
    """Matched-amplitude correlation over grid shifts; lags in seconds.

    The score is the least-squares matched pulse amplitude (relative to the
    best shift), so a small bump cannot masquerade as an alignment candidate.
    A shift only scores when the *entire* pulse support lands on present,
    unsaturated samples -- gap/rail regions are never interpolated over.
    Shifts are zero-padded (never wrapped) so the template cannot wrap around
    the capture ends.
    """
    dt = float(np.median(np.diff(grid)))
    p_grid_vals = _resample_on_grid(pulse_t, pulse, pmask, grid)
    p_valid = np.isfinite(p_grid_vals)
    p = np.where(p_valid, p_grid_vals, 0.0)
    s = np.where(smask, sig, 0.0)
    support_idx = np.where(p_valid)[0]
    i0, i1 = int(support_idx[0]), int(support_idx[-1])
    support = int(p_valid.sum())
    pp = float(np.dot(p[p_valid], p[p_valid])) or 1e-12
    n = len(grid)
    lags_list = []
    scores_list = []
    # shift k places the template's time zero at grid[k] (k*dt seconds)
    for k in range(n):
        shifted = np.zeros(n)
        shifted_valid = np.zeros(n, dtype=bool)
        lo = k + i0
        hi = k + i1 + 1
        if lo < 0 or hi > n:
            continue  # full template support must be inside the grid
        shifted[lo:hi] = p[i0:i1 + 1]
        shifted_valid[lo:hi] = p_valid[i0:i1 + 1]
        overlap = smask & shifted_valid
        if int(overlap.sum()) < support:
            continue  # template support touches a gap/rail: shift invalid
        sv = s[overlap]
        pv = shifted[overlap]
        lags_list.append(k * dt)
        scores_list.append(float(np.dot(sv, pv) / pp))
    lags = np.array(lags_list) if lags_list else np.array([0.0])
    scores = np.array(scores_list) if scores_list else np.array([0.0])
    top = scores.max() if scores.size else 0.0
    if top > 0:
        scores = scores / top
    return lags, scores


def _local_peaks(scores: np.ndarray) -> list[int]:
    peaks = []
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] >= scores[i + 1]:
            peaks.append(i)
    return peaks


def window_indices(times: np.ndarray, window: dict) -> tuple[int, int]:
    """Half-open [start, end) index range for a versioned measurement window."""
    lo = float(window["start_s"])
    hi = float(window["end_s"])
    idx = np.where((times >= lo) & (times < hi))[0]
    if idx.size == 0:
        return 0, 0
    return int(idx[0]), int(idx[-1]) + 1


def measure_window(times: np.ndarray, voltages: np.ndarray, mask: np.ndarray,
                   window: dict, regions: list[dict] | None = None) -> dict:
    """Peak-to-peak amplitude inside a window.

    Only present, unsaturated points count. Any saturated region that overlaps
    the window excludes the measurement; gaps do too when they cover too much
    of it. Invalid points are never interpolated.
    """
    lo_i, hi_i = window_indices(times, window)
    if hi_i <= lo_i:
        return {"included": False, "reason_code": "window-no-samples",
                "valid_fraction": 0.0}
    sub_mask = mask[lo_i:hi_i]
    total = hi_i - lo_i
    valid_fraction = float(sub_mask.sum()) / total
    sat_overlap = [r for r in (regions or [])
                   if r["code"].startswith("saturation")
                   and r["start_index"] < hi_i and r["end_index"] > lo_i]
    if sat_overlap:
        return {"included": False,
                "reason_code": "window-contains-saturation",
                "valid_fraction": valid_fraction,
                "start_index": lo_i, "end_index": hi_i,
                "window_version": window["version"],
                "regions": sat_overlap}
    if sub_mask.sum() == 0:
        return {"included": False, "reason_code": "window-all-invalid",
                "valid_fraction": valid_fraction,
                "regions": _sub_regions(mask, lo_i, hi_i),
                "start_index": lo_i, "end_index": hi_i,
                "window_version": window["version"]}
    v = voltages[lo_i:hi_i][sub_mask]
    vpp = float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).any() else None
    result = {
        "included": valid_fraction >= MIN_VALID_FRACTION,
        "valid_fraction": valid_fraction,
        "vpp": vpp,
        "start_index": lo_i, "end_index": hi_i,
        "window_version": window["version"],
    }
    if not result["included"]:
        result["reason_code"] = "window-too-much-invalid"
    return result


def _sub_regions(mask, lo_i, hi_i) -> list[dict]:
    regions = []
    i = lo_i
    while i < hi_i:
        if mask[i]:
            i += 1
            continue
        start = i
        while i < hi_i and not mask[i]:
            i += 1
        regions.append({"start_index": start, "end_index": i,
                        "length": i - start})
    return regions
