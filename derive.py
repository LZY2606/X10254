"""Derivation rules for the waveform overlay review tool.

Hard rules encoded here (also emitted as machine-readable events/ids):
  R1 时间轴不等间隔时不用索引相减 — time deltas always come from actual
      timestamps, never from sample indices.
  R2 缺样与饱和区间不进入插值 — missing/saturated intervals never feed
      interpolation; queries landing in them yield gaps, not fabricated data.
  R3 对齐保留多峰互相关候选 — alignment keeps every legitimate cross
      -correlation peak above threshold; nothing is auto-picked at import.

All domain quantities carry {"value", "unit"}; normalization happens only in
this derived layer. Unconvertible values stay in place with diagnostics.
"""
import hashlib
import json

import numpy as np

TIME_TO_NS = {"ns": 1.0, "ps": 1e-3, "us": 1e3, "s": 1e9}
VOLT_TO_V = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}

MAX_ALIGNMENT_CANDIDATES = 3


def source_hash(raw):
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def qty_to_ns(q):
    if isinstance(q, dict) and q.get("unit") in TIME_TO_NS:
        v = q.get("value")
        if isinstance(v, (int, float)):
            return v * TIME_TO_NS[q["unit"]]
    return None


def normalize_segment(raw):
    """Raw representation -> canonical (ns, V). Returns (norm, diagnostics).

    norm is None when a required field cannot be converted; the raw record is
    left untouched and the diagnostics explain why.
    """
    diags = []
    t = raw.get("t") or {}
    v = raw.get("v") or {}
    if t.get("unit") not in TIME_TO_NS:
        diags.append({
            "code": "unit.unconvertible", "field": "t",
            "unit": t.get("unit"),
            "message": "time unit %r cannot be converted to ns; raw values kept in place" % t.get("unit"),
        })
        return None, diags
    if v.get("unit") not in VOLT_TO_V:
        diags.append({
            "code": "unit.unconvertible", "field": "v",
            "unit": v.get("unit"),
            "message": "voltage unit %r cannot be converted to V; raw values kept in place" % v.get("unit"),
        })
        return None, diags
    atten_q = raw.get("probe_atten") or {}
    atten = atten_q.get("value", 1) if isinstance(atten_q.get("value", 1), (int, float)) else 1
    t_ns = [x * TIME_TO_NS[t["unit"]] for x in t.get("values", [])]
    v_v = [x * VOLT_TO_V[v["unit"]] * atten for x in v.get("values", [])]
    flags = raw.get("flags") or {}
    excluded = sorted(set(flags.get("missing", [])) | set(flags.get("saturated", [])))
    deskew_ns = qty_to_ns(raw.get("deskew")) or 0.0
    t_ns = [x - deskew_ns for x in t_ns]
    return {
        "t": t_ns,
        "v": v_v,
        "excluded": excluded,
        "deskew_ns": deskew_ns,
        "unit_basis": {"t": "ns", "v": "V"},
    }, diags


def time_deltas(t):
    """R1: deltas from actual timestamps — safe for non-uniform timebases."""
    return [t[i + 1] - t[i] for i in range(len(t) - 1)]


def interp_excluding(t, v, excluded, tq):
    """R2: linear interpolation that refuses to bridge excluded samples.

    Any query point whose bracketing samples include a missing/saturated
    index yields None (a gap), never a fabricated value.
    """
    excluded = set(excluded)
    out = []
    n = len(t)
    for x in tq:
        if x < t[0] or x > t[-1]:
            out.append(None)
            continue
        i = int(np.searchsorted(t, x, side="right")) - 1
        i = max(0, min(i, n - 2))
        if i in excluded or (i + 1) in excluded:
            out.append(None)
            continue
        t0, t1 = t[i], t[i + 1]
        if t1 == t0:
            out.append(v[i])
        else:
            frac = (x - t0) / (t1 - t0)
            out.append(v[i] + frac * (v[i + 1] - v[i]))
    return out


def _common_grid(ref, seg):
    t0 = max(ref["t"][0], seg["t"][0])
    t1 = min(ref["t"][-1], seg["t"][-1])
    if t1 <= t0:
        return None
    deltas = time_deltas(ref["t"])
    dt = float(np.median(deltas)) if deltas else 0.0
    if dt <= 0:
        return None
    return np.arange(t0, t1 + dt * 0.5, dt), dt


def align_candidates(ref, seg, threshold=0.6, max_candidates=MAX_ALIGNMENT_CANDIDATES):
    """R3: keep every cross-correlation peak above threshold (multi-peak).

    Returns a list of {"lag_ns", "score"} sorted by score, best first.
    """
    grid = _common_grid(ref, seg)
    if grid is None:
        return []
    tg, dt = grid
    rv = interp_excluding(ref["t"], ref["v"], ref["excluded"], tg)
    sv = interp_excluding(seg["t"], seg["v"], seg["excluded"], tg)
    rv = np.array([np.nan if x is None else x for x in rv])
    sv = np.array([np.nan if x is None else x for x in sv])
    valid = ~np.isnan(rv) & ~np.isnan(sv)
    if valid.sum() < 8:
        return []
    r = rv[valid] - np.nanmean(rv[valid])
    max_lag = int(round((tg[-1] - tg[0]) / dt)) // 2
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = np.full_like(rv, np.nan)
        if lag >= 0:
            shifted[lag:] = sv[: len(sv) - lag]
        else:
            shifted[: lag] = sv[-lag:]
        mask = ~np.isnan(rv) & ~np.isnan(shifted)
        if mask.sum() < 8:
            scores.append(0.0)
            continue
        a = rv[mask] - rv[mask].mean()
        b = shifted[mask] - shifted[mask].mean()
        denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
        scores.append(float((a * b).sum() / denom) if denom > 0 else 0.0)
    scores = np.array(scores)
    lags = np.arange(-max_lag, max_lag + 1)
    peaks = []
    for i in range(1, len(scores) - 1):
        if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1] and scores[i] >= threshold:
            peaks.append((float(scores[i]), int(lags[i])))
    peaks.sort(reverse=True)
    return [{"lag_ns": round(lag * dt, 6), "score": round(score, 6)}
            for score, lag in peaks[:max_candidates]]


def window_check(trigger_ns, window):
    """Measurement-window membership with versioned boundary inclusivity."""
    start = window["start_ns"]
    end = window["end_ns"]
    lo_ok = trigger_ns >= start if window.get("start_inclusive") else trigger_ns > start
    hi_ok = trigger_ns <= end if window.get("end_inclusive", True) else trigger_ns < end
    if lo_ok and hi_ok:
        return True, None
    if trigger_ns == start and not window.get("start_inclusive"):
        return False, "trigger at window boundary (start, exclusive in this window version)"
    if trigger_ns == end and not window.get("end_inclusive", True):
        return False, "trigger at window boundary (end, exclusive in this window version)"
    return False, "trigger outside measurement window [%.3f, %.3f] ns" % (start, end)


def compute_derived(state, case_meta):
    """Compute the full derived layer for a replayed branch state.

    Every item carries a stable id, its dependency versions and its source
    chain so the UI can display identity and the invalidation scope can be
    computed from dependencies rather than by global replacement.
    """
    params = state["params"]
    mw_version = state["mw_version"]
    window = case_meta["meas_windows"][mw_version]
    threshold = params.get("align_threshold", 0.6)
    expected_trigger = params.get("expected_trigger_ns", 0.0)

    norms = {}
    diagnostics = []
    exclusions = []
    for sid, seg in state["segments"].items():
        raw = seg["raw"]
        norm, diags = normalize_segment(raw)
        for d in diags:
            diagnostics.append({"id": "diag:%s:%s" % (seg["key"], d["field"]),
                                "segment": seg["key"], **d})
        if norm is None:
            exclusions.append({
                "id": "excl:%s" % seg["key"],
                "segment": seg["key"],
                "reason": "unconvertible units; raw record preserved",
                "rule": "unit.unconvertible",
                "deps": ["seg:%s@%s" % (seg["key"], seg["hash"])],
            })
        else:
            norms[sid] = norm

    alignment = {}
    ref_id = state["order"][0] if state["order"] else None
    for sid in state["order"]:
        seg = state["segments"][sid]
        if sid not in norms:
            continue
        deps = ["seg:%s@%s" % (seg["key"], seg["hash"]),
                "param:align_threshold@%s" % threshold]
        cands = []
        if sid != ref_id and ref_id in norms:
            ref_seg = state["segments"][ref_id]
            deps.append("seg:%s@%s" % (ref_seg["key"], ref_seg["hash"]))
            for i, c in enumerate(align_candidates(norms[ref_id], norms[sid], threshold), 1):
                cid = "cand:%s#%d" % (seg["key"], i)
                adj = state["adjudications"].get(cid)
                cands.append({
                    "id": cid,
                    "lag_ns": c["lag_ns"],
                    "score": c["score"],
                    "status": adj["decision"] if adj else "pending",
                    "decided_by": adj["event_id"] if adj else None,
                    "unit": "ns",
                })
        alignment[sid] = {
            "id": "align:%s" % seg["key"],
            "segment": seg["key"],
            "candidates": cands,
            "deps": deps,
            "sources": [seg["key"]] + ([state["segments"][ref_id]["key"]] if sid != ref_id and ref_id in norms else []),
        }

    jitter = {}
    windows = {}
    for sid in state["order"]:
        seg = state["segments"][sid]
        raw = seg["raw"]
        trig_ns = qty_to_ns(raw.get("trigger_ts"))
        if trig_ns is None:
            diagnostics.append({
                "id": "diag:%s:trigger_ts" % seg["key"], "segment": seg["key"],
                "code": "unit.unconvertible", "field": "trigger_ts",
                "unit": (raw.get("trigger_ts") or {}).get("unit"),
                "message": "trigger timestamp cannot be converted to ns",
            })
            continue
        jitter[sid] = {
            "id": "jit:%s" % seg["key"],
            "segment": seg["key"],
            "value_ns": round(trig_ns - expected_trigger, 6),
            "unit": "ns",
            "deps": ["seg:%s@%s" % (seg["key"], seg["hash"]),
                     "param:expected_trigger_ns@%s" % expected_trigger],
            "sources": [seg["key"]],
        }
        ok, reason = window_check(trig_ns, window)
        windows[sid] = {
            "id": "win:%s@%s" % (seg["key"], mw_version),
            "segment": seg["key"],
            "window_version": mw_version,
            "included": ok,
            "reason": reason,
            "deps": ["seg:%s@%s" % (seg["key"], seg["hash"]), "mw@%s" % mw_version],
            "sources": [seg["key"], mw_version],
        }
        if not ok:
            exclusions.append({
                "id": "excl:%s:window" % seg["key"],
                "segment": seg["key"],
                "reason": reason,
                "rule": "window.membership",
                "deps": ["seg:%s@%s" % (seg["key"], seg["hash"]), "mw@%s" % mw_version],
            })

    return {
        "alignment": alignment,
        "jitter": jitter,
        "windows": windows,
        "exclusions": exclusions,
        "diagnostics": diagnostics,
        "normalized": {sid: {"t": n["t"], "v": n["v"], "excluded": n["excluded"]}
                       for sid, n in norms.items()},
    }
