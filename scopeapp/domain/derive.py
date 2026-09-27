"""Derived layer: normalisation, multi-peak alignment and measurement windows.

The rules below are the versioned derivation rules of the tool:

* ``R-TIME-001``: with a non-uniform time axis, physical time differences must
  be computed from timestamps; subtracting sample indices is forbidden.
* ``R-INTERP-001``: missing samples and saturated intervals never enter
  interpolation; a resampled point bracketed by or adjacent to such an
  interval is invalid.
* ``R-ALIGN-003``: alignment keeps every legal peak of the cross-correlation
  as a candidate; ambiguous cases are not collapsed until a ruling.
* ``R-DESKEW-001``: channel deskew is applied as a coordinate offset and
  provenance keeps both raw and corrected event times.
* ``R-TRIG-001``: a trigger marker on a segment boundary leaves no
  pre-trigger context; the segment is excluded from alignment/jitter with a
  machine readable reason.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .units import (
    ConversionDiagnostic,
    parse_probe_attenuation,
    time_quantity,
    voltage_quantity,
)

RULES = {
    "R-TIME-001": "non-uniform time axis: use timestamp deltas, never index subtraction",
    "R-INTERP-001": "missing samples and saturated intervals do not enter interpolation",
    "R-ALIGN-003": "alignment keeps all legal cross-correlation peaks as candidates",
    "R-DESKEW-001": "channel deskew applied as a coordinate offset with provenance",
    "R-TRIG-001": "trigger marker on boundary: segment excluded, reason retained",
    "R-WIN-001": "window statistics use only valid in-window samples",
}

PROMINENCE = 0.80
MIN_PEAK_SEPARATION_NS = 12.0


def _stable(kind: str, *parts: Any) -> str:
    return ".".join([kind, *[str(part) for part in parts]])


def convert_channel(raw: dict, diagnostics: list[ConversionDiagnostic]) -> dict:
    """Normalise one channel calibration while preserving raw representations."""
    cid = raw["id"]
    gain_conv = voltage_quantity(raw.get("v_per_code"), _stable("ch", cid), "v_per_code")
    if gain_conv.diagnostic:
        diagnostics.append(gain_conv.diagnostic)
    attenuation = parse_probe_attenuation(
        raw.get("probe_attenuation"), _stable("ch", cid), diagnostics
    )
    deskew_conv = time_quantity(raw.get("deskew"), _stable("ch", cid), "deskew")
    if deskew_conv.diagnostic:
        diagnostics.append(deskew_conv.diagnostic)
    return {
        "id": cid,
        "label": raw.get("label", cid),
        "v_per_code_si": gain_conv.si_value,
        "v_per_code": gain_conv.to_dict(),
        "probe_attenuation": attenuation if attenuation is not None else 1.0,
        "probe_attenuation_raw": raw.get("probe_attenuation"),
        "probe_attenuation_usable": attenuation is not None,
        "deskew_si": deskew_conv.si_value if deskew_conv.ok else 0.0,
        "deskew": deskew_conv.to_dict(),
        "code_min": raw.get("code_min", -2048),
        "code_max": raw.get("code_max", 2047),
    }


def time_axis(seg: dict, acquisition: dict,
              diagnostics: list[ConversionDiagnostic]) -> np.ndarray | None:
    """Build physical time axis (seconds, relative to t0) from the segment.

    Uniform segments use the acquisition sample interval.  Non-uniform
    segments carry explicit ``times`` (R-TIME-001).  A segment that cannot be
    placed on a physical axis returns ``None`` and records a diagnostic.
    """
    n = len(seg["samples"])
    times_raw = seg.get("times")
    if times_raw is not None:
        sid = seg["id"]
        converted = [time_quantity(t, _stable("seg", sid, "t", i), "time")
                     for i, t in enumerate(times_raw)]
        for conv in converted:
            if conv.diagnostic:
                diagnostics.append(conv.diagnostic)
        if not all(conv.ok for conv in converted) or len(converted) != n:
            return None
        axis = np.array([conv.si_value for conv in converted], dtype=float)
        if np.any(np.diff(axis) <= 0):
            diagnostics.append(ConversionDiagnostic(
                seg["id"], "times", "non_monotonic_time",
                "explicit time axis must be strictly increasing", seg.get("times"),
            ))
            return None
        return axis
    interval_conv = time_quantity(
        acquisition.get("sample_interval"),
        _stable("acq", acquisition.get("id", "acq")), "sample_interval",
    )
    if interval_conv.diagnostic:
        diagnostics.append(interval_conv.diagnostic)
    if not interval_conv.ok or interval_conv.si_value is None:
        return None
    return np.arange(n, dtype=float) * interval_conv.si_value


def voltage_series(seg: dict, channel: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(voltage, valid_mask, saturated_mask)`` in volts at the scope input.

    Raw codes that are ``None`` are missing samples.  Codes at the ADC rails
    are saturated.  Neither class enters interpolation later (R-INTERP-001).
    """
    raw_codes = seg["samples"]
    samples = np.array([np.nan if c is None else float(c) for c in raw_codes])
    missing = np.array([c is None for c in raw_codes])
    saturated = np.zeros_like(samples, dtype=bool)
    usable_gain = channel["v_per_code_si"] is not None
    vpc = channel["v_per_code_si"] if usable_gain else 1.0
    atten = channel["probe_attenuation"] if channel["probe_attenuation_usable"] else 1.0
    if usable_gain:
        saturated = (~missing) & (
            (samples <= channel["code_min"]) | (samples >= channel["code_max"])
        )
    voltage = samples * vpc * atten
    valid = (~missing) & (~saturated) & usable_gain
    return voltage, valid, saturated


def trigger_offset_seconds(seg: dict, diagnostics: list[ConversionDiagnostic]) -> float | None:
    conv = time_quantity(seg.get("trigger_offset"), _stable("seg", seg["id"]),
                         "trigger_offset")
    if conv.diagnostic:
        diagnostics.append(conv.diagnostic)
    return conv.si_value if conv.ok else None


def interpolable_grid(times: np.ndarray, grid: np.ndarray,
                      valid: np.ndarray, saturated: np.ndarray) -> np.ndarray:
    """Boolean mask of ``grid`` points safe to interpolate (R-INTERP-001).

    A grid point is interpolable only when its two bracketing raw samples are
    both valid, they are adjacent raw indices, and the point is not inside a
    saturated or missing interval.  This deliberately performs no fill.
    """
    ok = np.zeros_like(grid, dtype=bool)
    n = len(times)
    order = np.searchsorted(times, grid, side="left")
    raw_good = valid & ~saturated
    for k, g in enumerate(grid):
        j = int(order[k])
        if j <= 0 or j >= n:
            continue
        if not (raw_good[j - 1] and raw_good[j]):
            continue
        t0, t1 = times[j - 1], times[j]
        if t1 <= t0:
            continue
        ok[k] = True
    return ok


def resample(times: np.ndarray, voltage: np.ndarray, raw_good: np.ndarray,
             grid: np.ndarray) -> np.ndarray:
    """Linear resampling; points failing R-INTERP-001 are ``np.nan``."""
    out = np.full_like(grid, np.nan, dtype=float)
    ok = interpolable_grid(times, grid, raw_good, np.zeros_like(raw_good))
    usable = np.where(raw_good, voltage, np.nan)
    for k, g in enumerate(grid):
        if not ok[k]:
            continue
        j = int(np.searchsorted(times, g, side="left"))
        t0, t1 = times[j - 1], times[j]
        frac = (g - t0) / (t1 - t0)
        out[k] = usable[j - 1] + frac * (usable[j] - usable[j - 1])
    return out


def correlation_lags(reference: np.ndarray, target: np.ndarray,
                     grid_step: float) -> tuple[np.ndarray, np.ndarray]:
    """Masked normalised cross-correlation; lag index -> seconds, no index math.

    NaN samples (missing / saturated neighbours per R-INTERP-001) are masked
    pairwise; a lag with fewer than eight valid overlaps is not scored.
    """
    ref = reference - np.nanmean(reference)
    tgt = target - np.nanmean(target)
    max_lag = len(ref) - 1
    scores: list[float] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = ref[-lag:], target[: lag]
        elif lag > 0:
            a, b = ref[:-lag], target[lag:]
        else:
            a, b = ref, target
        both = ~(np.isnan(a) | np.isnan(b))
        if both.sum() < 8:
            scores.append(np.nan)
            continue
        aa, bb = a[both], b[both]
        denom = math.sqrt(float(np.dot(aa, aa)) * float(np.dot(bb, bb)))
        scores.append(float(np.dot(aa, bb) / denom) if denom > 0 else np.nan)
    corr = np.array(scores, dtype=float)
    lags_s = np.arange(-max_lag, max_lag + 1, dtype=float) * grid_step
    return lags_s, corr


def refine_peak(lags: np.ndarray, corr: np.ndarray, index: int) -> tuple[float, float]:
    """Parabolic sub-sample refinement; returns (lag_seconds, score)."""
    if index <= 0 or index >= len(corr) - 1:
        return float(lags[index]), float(corr[index])
    y0, y1, y2 = corr[index - 1], corr[index], corr[index + 1]
    denom = (y0 - 2 * y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    step = float(lags[1] - lags[0])
    return float(lags[index] + delta * step), float(y1)


def find_peak_candidates(lags: np.ndarray, corr: np.ndarray) -> list[dict]:
    """Return all legal correlation peaks (R-ALIGN-003, never just the argmax)."""
    valid = ~np.isnan(corr)
    idx = np.where(valid)[0]
    local: list[int] = []
    for k in idx:
        if k - 1 not in idx or k + 1 not in idx:
            continue
        if corr[k] > corr[k - 1] and corr[k] >= corr[k + 1]:
            local.append(int(k))
    if not local:
        best = int(np.nanargmax(corr))
        local = [best]
    top = max(corr[k] for k in local)
    strong = sorted(
        (k for k in local if corr[k] >= PROMINENCE * top),
        key=lambda k: -corr[k],
    )
    pruned: list[int] = []
    for k in strong:
        if all(abs(lags[k] - lags[j]) >= MIN_PEAK_SEPARATION_NS * 1e-9
               for j in pruned):
            pruned.append(k)
    candidates = []
    for rank, k in enumerate(pruned):
        lag_s, score = refine_peak(lags, corr, k)
        candidates.append({"rank": rank, "lag_s": lag_s, "score": score})
    return candidates


def window_metrics(times_aligned: np.ndarray, voltage: np.ndarray,
                   valid: np.ndarray, window: dict) -> dict:
    """Statistics over a measurement window using valid samples only (R-WIN-001)."""
    start = float(window["start_s"])
    end = float(window["end_s"])
    inside = (times_aligned >= start) & (times_aligned <= end)
    usable = inside & valid
    total = int(inside.sum())
    n = int(usable.sum())
    result = {
        "window_id": window["id"],
        "start_s": start,
        "end_s": end,
        "samples_in_window": total,
        "samples_used": n,
        "valid_fraction": (n / total) if total else 0.0,
        "vpp_si": None,
        "clipped": False,
        "usable": total > 0 and n == total,
    }
    if n:
        values = voltage[usable]
        result["vpp_si"] = float(np.max(values) - np.min(values))
        result["vpp_unit"] = "V"
    return result


def _val(value_id: str, value: Any, unit: str | None,
         basis: str | None = None, **extra: Any) -> dict:
    """Wrap a display value with a stable identity."""
    out = {"id": value_id, "value": value, "unit": unit, "basis": basis}
    out.update(extra)
    return out


def derive_snapshot(material: dict, revisions: dict[str, dict],
                    interpretation: dict) -> dict:
    """Derive a read-model snapshot from source material and rulings.

    ``revisions`` maps a parameter address (``channel:<id>:<field>``) to the
    winning revised parameter; ``interpretation`` maps candidate ids to a
    decision (``accept`` / ``reject``).  Nothing here mutates source data.
    """
    diagnostics: list[ConversionDiagnostic] = []
    acquisition = material["acquisition"]
    channels_raw = {ch["id"]: ch for ch in material["channels"]}
    channels: dict[str, dict] = {}
    for ch in material["channels"]:
        converted = convert_channel(ch, diagnostics)
        for key in ("deskew", "v_per_code", "probe_attenuation"):
            address = f"channel:{ch['id']}:{key}"
            if address in revisions:
                revised = revisions[address]
                if key == "deskew":
                    conv = time_quantity(revised["value"], address, "deskew")
                    if conv.diagnostic:
                        diagnostics.append(conv.diagnostic)
                    converted["deskew_si"] = conv.si_value if conv.ok else 0.0
                    converted["deskew"] = conv.to_dict()
                    converted["deskew_revised_by"] = revised["event_id"]
                elif key == "v_per_code":
                    conv = voltage_quantity(revised["value"], address, "v_per_code")
                    if conv.diagnostic:
                        diagnostics.append(conv.diagnostic)
                    converted["v_per_code_si"] = conv.si_value
                    converted["v_per_code"] = conv.to_dict()
                    converted["gain_revised_by"] = revised["event_id"]
                elif key == "probe_attenuation":
                    converted["probe_attenuation"] = float(revised["value"])
                    converted["probe_attenuation_raw"] = revised["value"]
        channels[ch["id"]] = converted

    segments_by_channel: dict[str, list[dict]] = {}
    deps: dict[str, set[str]] = {}
    for cid, channel in channels.items():
        converted_segments: list[dict] = []
        for seg in material["segments"]:
            sid = seg["id"]
            seg_id = _stable("seg", cid, sid)
            trig = trigger_offset_seconds(seg, diagnostics)
            axis = time_axis(seg, acquisition, diagnostics)
            voltage, valid, saturated = voltage_series(seg, channel)
            n_missing = int((~valid).sum() - saturated.sum())
            n_saturated = int(saturated.sum())
            n = len(seg["samples"])
            boundary = trig is not None and (trig <= 0.0 or (
                axis is not None and trig >= float(axis[-1])))
            converted_segments.append({
                "id": seg_id,
                "segment_id": sid,
                "channel_id": cid,
                "trigger_offset_si": trig,
                "trigger_offset_raw": seg.get("trigger_offset"),
                "times_si": None if axis is None else axis.tolist(),
                "times_raw": seg.get("times"),
                "uniform": seg.get("times") is None,
                "voltage": voltage.tolist(),
                "valid": valid.tolist(),
                "saturated": saturated.tolist(),
                "n_samples": n,
                "n_missing": int(sum(c is None for c in seg["samples"])),
                "n_saturated": n_saturated,
                "boundary_trigger": bool(boundary),
                "candidates": [],
                "alignment": None,
                "excluded": False,
                "exclusion_reasons": [],
                "window_metrics": [],
            })
            deps[seg_id] = {
                f"channel:{cid}:deskew",
                f"channel:{cid}:v_per_code",
                f"channel:{cid}:probe_attenuation",
                f"segment:{sid}:samples",
                f"segment:{sid}:times",
                f"segment:{sid}:trigger_offset",
            }
        segments_by_channel[cid] = converted_segments

    interval_conv = time_quantity(acquisition.get("sample_interval"), "acq",
                                  "sample_interval")
    grid_step = interval_conv.si_value if interval_conv.ok else None

    channel_summaries: dict[str, dict] = {}
    for cid, channel in channels.items():
        segs = segments_by_channel[cid]
        usable = [s for s in segs if not s["boundary_trigger"]
                  and s["times_si"] is not None and not all(~np.array(s["valid"]))]
        for seg in segs:
            if seg["boundary_trigger"]:
                seg["excluded"] = True
                seg["exclusion_reasons"].append({
                    "code": "trigger_at_boundary", "rule": "R-TRIG-001",
                    "message": "trigger marker sits on a segment boundary; no pre/post context",
                })
            elif seg["times_si"] is None:
                seg["excluded"] = True
                seg["exclusion_reasons"].append({
                    "code": "time_axis_unusable", "rule": "R-TIME-001",
                    "message": "no physical time axis could be constructed",
                })
            elif all(~np.array(seg["valid"])):
                seg["excluded"] = True
                seg["exclusion_reasons"].append({
                    "code": "no_valid_samples", "rule": "R-INTERP-001",
                    "message": "every sample is missing or saturated",
                })

        if not usable or grid_step is None:
            channel_summaries[cid] = {"id": cid, "jitter": None,
                                      "reference_segment": None}
            continue

        reference = max(usable, key=lambda s: (int(np.sum(s["valid"])), s["segment_id"]))
        n_grid = max(s["n_samples"] for s in usable)
        grid = np.arange(n_grid, dtype=float) * grid_step
        ref_grid = resample(
            np.array(reference["times_si"]), np.array(reference["voltage"]),
            np.array(reference["valid"]), grid,
        )
        reference["aligned_grid_s"] = grid.tolist()
        reference["aligned_voltage"] = ref_grid.tolist()
        reference["alignment"] = {"candidate_id": None, "lag_si": 0.0,
                                  "event_offset_corrected_si": -channel["deskew_si"],
                                  "reference": True}

        for seg in usable:
            tgt = resample(
                np.array(seg["times_si"]), np.array(seg["voltage"]),
                np.array(seg["valid"]), grid,
            )
            lags, corr = correlation_lags(ref_grid, tgt, grid_step)
            peaks = find_peak_candidates(lags, corr)
            candidates = []
            for peak in peaks:
                cand_id = _stable("cand", cid, seg["segment_id"],
                                  f"p{peak['rank'] + 1:02d}")
                decision = interpretation.get(cand_id)
                candidates.append({
                    "id": cand_id,
                    "rank": peak["rank"],
                    "lag_si": _val(_stable("v", cand_id, "lag"),
                                   peak["lag_s"], "s",
                                   "aligned coordinate shift vs reference"),
                    "score": _val(_stable("v", cand_id, "score"),
                                  peak["score"], None,
                                  "normalised cross-correlation peak"),
                    "decision": decision,
                    "decision_event_id": interpretation.get(cand_id + ":event"),
                })
            seg["candidates"] = candidates
            accepted = [c for c in candidates if c["decision"] == "accept"]
            if seg is reference:
                pass
            elif len(accepted) == 1:
                chosen = accepted[0]
                lag_si = chosen["lag_si"]["value"]
                seg["alignment"] = {
                    "candidate_id": chosen["id"],
                    "lag_si": lag_si,
                    "event_offset_corrected_si": lag_si - channel["deskew_si"],
                    "reference": False,
                }
                seg["aligned_grid_s"] = grid.tolist()
                seg["aligned_voltage"] = tgt.tolist()
            elif len(candidates) > 1 and not accepted:
                seg["excluded"] = True
                seg["exclusion_reasons"].append({
                    "code": "ambiguous_multipeak_pending", "rule": "R-ALIGN-003",
                    "message": "multiple legal correlation peaks; a ruling is required",
                })
            elif candidates:
                chosen = candidates[0]
                lag_si = chosen["lag_si"]["value"]
                seg["alignment"] = {
                    "candidate_id": chosen["id"],
                    "lag_si": lag_si,
                    "event_offset_corrected_si": lag_si - channel["deskew_si"],
                    "reference": False,
                }
                seg["aligned_grid_s"] = grid.tolist()
                seg["aligned_voltage"] = tgt.tolist()
            else:
                seg["excluded"] = True
                seg["exclusion_reasons"].append({
                    "code": "no_correlation_peak", "rule": "R-ALIGN-003",
                    "message": "no legal cross-correlation peak",
                })

            latest_windows: dict[str, dict] = {}
            for win in material.get("windows", []):
                current = latest_windows.get(win["id"])
                if current is None or int(win.get("version", 1)) > int(
                        current.get("version", 1)):
                    latest_windows[win["id"]] = win
            for seg in segs:
                if seg["times_si"] is None:
                    continue
                times = np.array(seg["times_si"])
                voltage = np.array(seg["voltage"])
                valid_mask = np.array(seg["valid"], dtype=bool)
                if seg["alignment"] and not seg["excluded"]:
                    shifted = times - seg["alignment"]["lag_si"]
                else:
                    shifted = times
                for win in latest_windows.values():
                    metrics = window_metrics(shifted, voltage, valid_mask, win)
                    metrics["id"] = _stable("winm", cid, seg["segment_id"],
                                            win["id"], f"v{win.get('version', 1)}")
                    metrics["window_version"] = int(win.get("version", 1))
                    metrics["vpp"] = _val(
                        _stable("v", metrics["id"], "vpp"),
                        metrics.pop("vpp_si"), "V",
                        "peak-to-peak over valid in-window samples",
                    )
                    seg["window_metrics"].append(metrics)
                    deps[metrics["id"]] = {
                        seg["id"],
                        f"window:{win['id']}:v{win.get('version', 1)}",
                        "R-WIN-001",
                    }

            for seg in segs:
                if seg["candidates"]:
                    for candidate in seg["candidates"]:
                        deps[candidate["id"]] = {seg["id"], "R-ALIGN-003"}

            aligned_segments = [s for s in usable
                                if s["alignment"] is not None and not s["excluded"]]
            events = [s["alignment"]["event_offset_corrected_si"]
                      for s in aligned_segments]
            jitter = None
            if len(events) >= 2:
                jitter_si = float(np.std(np.array(events), ddof=0))
                jitter = {
                    "id": _stable("jitter", cid),
                    "std": _val(_stable("v", "jitter", cid, "std"),
                                jitter_si, "s",
                                "std of deskew-corrected aligned event offsets"),
                    "members": [s["id"] for s in aligned_segments],
                }
                deps[_stable("jitter", cid)] = {
                    s["id"] for s in aligned_segments
                } | {f"channel:{cid}:deskew"}

            channel_summaries[cid] = {
                "id": _stable("chanview", cid),
                "channel_id": cid,
                "reference_segment": reference["id"],
                "grid_step": _val(_stable("v", "grid", cid), grid_step, "s",
                                  "common resample grid interval"),
                "jitter": jitter,
            }

    dependencies = {key: sorted(value) for key, value in deps.items()}
    return {
        "schema": "scopeapp.snapshot/v1",
        "material_id": material["id"],
        "rule_versions": dict(RULES),
        "channels": channels,
        "segments": {cid: segs for cid, segs in segments_by_channel.items()},
        "channel_views": channel_summaries,
        "windows": material.get("windows", []),
        "diagnostics": [d.to_dict() for d in diagnostics],
        "dependencies": dependencies,
    }


def invalidation_scope(dependencies: dict[str, list[str]],
                       changed_sources: set[str]) -> set[str]:
    """Transitive closure: which derived ids become stale when sources change.

    Parameter or reference version changes invalidate only objects that
    depend (transitively) on them.  Old conclusions are never globally
    replaced silently.
    """
    stale: set[str] = set()
    frontier = set(changed_sources)
    while frontier:
        nxt: set[str] = set()
        for derived_id, requires in dependencies.items():
            if derived_id in stale:
                continue
            if any(req in stale or req in changed_sources for req in requires):
                if derived_id not in stale:
                    nxt.add(derived_id)
        if not nxt:
            break
        stale |= nxt
    return stale
