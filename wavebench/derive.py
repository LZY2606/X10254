"""Derived layer: calibration, multi-peak alignment, jitter, windows.

Source materials stay untouched; everything here is a pure function of the
folded interpretation (materials + parameter/rule versions + rulings).
Deterministic: same inputs -> same ids/order/conclusions.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .units import convert_seconds, convert_volts

JUDGMENT_RULES_VERSION = "rules-1.0.0"

# judgment code -> human rule statement (machine-readable counterpart is JSON)
RULE_STATEMENTS = {
    "R1": "时间轴不等间隔时不用索引相减：对齐平移以标定后的物理秒估计",
    "R2": "缺样区间不进入插值：缺口跨越点不产生重建样本",
    "R3": "饱和区间不进入插值与测量：钳位平台仅作标注",
    "R4": "对齐保留多峰互相关候选：无法唯一决定时全部合法候选进入裁定",
    "R5": "通道 deskew 在派生层应用：触发/特征时刻按各通道 deskew 修正",
    "R6": "触发位于记录边界时不纳入抖动分布",
    "R7": "测量窗口按版本生效：窗口修订只失效测量，不替换对齐结论",
    "D1": "时间单位无法转换：原始表示保留在来源层，仅派生层给出诊断",
    "D2": "导入表示损坏：原始字节保留，派生层跳过并附诊断",
    "R0": "重复事件对齐：以参考段标定秒网格做归一化互相关",
}

SAT_CODE_FRACTION_OF_FSR = 0.98
MIN_OVERLAP_FRACTION = 0.40
SECOND_PEAK_RATIO = 0.85
MIN_PEAK_SEPARATION_S = 4e-9
GRID_POINTS = 121
INVALID_FRACTION_LIMIT = 0.25
JITTER_PAD_FRACTION = 0.05


def hsid(prefix: str, token: str) -> str:
    return f"{prefix}_{hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]}"


def fnum(x: Any) -> float:
    return float(round(float(x), 9))


@dataclass
class SegmentView:
    seg_id: str
    channel: str
    reference: bool
    included: bool
    excluded_reasons: List[Dict[str, str]] = field(default_factory=list)
    judgments: List[Dict[str, Any]] = field(default_factory=list)
    original: Dict[str, Any] = field(default_factory=dict)
    timeaxis: Optional[Dict[str, Any]] = None
    waveform: Optional[Dict[str, Any]] = None
    align: Optional[Dict[str, Any]] = None
    jitter_ns: Optional[float] = None
    measurements: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DerivedState:
    items: Dict[str, Dict[str, Any]]
    segments: List[SegmentView]
    judgments: List[Dict[str, Any]]
    params: Dict[str, Any]
    windows: List[Dict[str, Any]]
    segment_order: List[str]
    selected_align: Dict[str, str]
    rulings: List[Dict[str, Any]]


def _jdg(seg_id: str, code: str, status: str, basis: str,
         affected: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": hsid("jdg", f"{seg_id}|{code}|{basis}"),
        "segment_id": seg_id,
        "code": code,
        "rules_version": JUDGMENT_RULES_VERSION,
        "status": status,  # accepted | rejected | diagnostic
        "basis": RULE_STATEMENTS[code],
        "affected_ids": affected,
        "evidence": evidence,
    }


def _calibrate_segment(mat: Dict[str, Any]) -> Tuple[SegmentView, Optional[Dict[str, Any]]]:
    seg = mat["data"]
    seg_id = seg["segment_id"]
    view = SegmentView(
        seg_id=seg_id,
        channel=seg.get("channel", "?"),
        reference=bool(seg.get("reference", False)),
        included=True,
        original=seg,
    )

    def exclude(code: str, diagnostic: str) -> None:
        view.included = False
        view.excluded_reasons.append({
            "judgment_id": hsid("jdg", f"{seg_id}|{code}|excl"),
            "code": code,
            "rules_version": JUDGMENT_RULES_VERSION,
            "reason": RULE_STATEMENTS[code],
            "diagnostic": diagnostic,
        })

    # --- timebase (original representation preserved on view.original) ---
    tb = seg.get("timebase", {})
    pts = seg.get("points", [])
    if tb.get("uniform"):
        start = convert_seconds(tb.get("start", {}))
        step = convert_seconds(tb.get("step", {}))
        if not (start.ok and step.ok):
            view.judgments.append(_jdg(seg_id, "D1", "diagnostic",
                                      RULE_STATEMENTS["D1"], [],
                                      {"start": tb.get("start"), "step": tb.get("step")}))
            exclude("D1", start.diagnostic or step.diagnostic)
            return view, None
        n = len(pts)
        raw_t = np.array([start.value + step.value * i for i in range(n)], dtype=float)
        t_unit = {"start": tb.get("start"), "step": tb.get("step")}
    else:
        stamps = tb.get("stamps", [])
        if len(stamps) != len(pts) or not stamps:
            view.judgments.append(_jdg(seg_id, "D2", "diagnostic",
                                      RULE_STATEMENTS["D2"], [],
                                      {"stamp_count": len(stamps), "point_count": len(pts)}))
            exclude("D2", "timestamp/point count mismatch")
            return view, None
        conv = [convert_seconds(s) for s in stamps]
        bad = [c for c in conv if not c.ok]
        if bad:
            view.judgments.append(_jdg(seg_id, "D1", "diagnostic",
                                      RULE_STATEMENTS["D1"], [],
                                      {"stamps": stamps}))
            exclude("D1", bad[0].diagnostic)
            return view, None
        raw_t = np.array([c.value for c in conv], dtype=float)
        t_unit = {"stamps": stamps}
        diffs = np.diff(raw_t)
        nonuniform = bool(np.any(np.abs(diffs - np.median(diffs)) > 0.05 * np.median(diffs)))
        if nonuniform:
            view.judgments.append(_jdg(
                seg_id, "R1", "accepted", RULE_STATEMENTS["R1"],
                [hsid("ta", seg_id), hsid("align", seg_id)],
                {"naive_index_lag_note": "lag 以物理秒求解；索引差不等于时间差",
                 "median_step_s": fnum(np.median(diffs))}))

    # --- trigger ---
    trig = convert_seconds(seg.get("trigger", {}))
    if not trig.ok:
        view.judgments.append(_jdg(seg_id, "D1", "diagnostic",
                                  RULE_STATEMENTS["D1"], [], {"trigger": seg.get("trigger")}))
        exclude("D1", trig.diagnostic)
        return view, None

    # --- vertical calibration: codes -> volts at probe tip ---
    gain = convert_volts(seg.get("vertical_gain", {}))
    offs = convert_volts(seg.get("vertical_offset", {"value": 0, "unit": "V"}))
    probe = seg.get("probe_attenuation", {"value": 1})
    if not (gain.ok and offs.ok):
        view.judgments.append(_jdg(seg_id, "D1", "diagnostic",
                                  RULE_STATEMENTS["D1"], [],
                                  {"gain": seg.get("vertical_gain")}))
        exclude("D1", gain.diagnostic or offs.diagnostic)
        return view, None
    atten = float(probe.get("value", 1))
    deskew = convert_seconds(seg.get("deskew", {"value": 0, "unit": "s"}))
    if not deskew.ok:
        view.judgments.append(_jdg(seg_id, "D1", "diagnostic",
                                  RULE_STATEMENTS["D1"], [],
                                  {"deskew": seg.get("deskew")}))
        exclude("D1", deskew.diagnostic)
        return view, None

    codes = np.array([np.nan if p is None else float(p) for p in pts], dtype=float)
    present = np.isfinite(codes)
    fsr = _full_scale_codes(seg)
    sat = present & (np.abs(codes) >= SAT_CODE_FRACTION_OF_FSR * fsr)
    saturated = bool(np.any(sat))
    missing_fraction = fnum(float(np.mean(~present))) if len(codes) else 1.0

    volts = (codes * gain.value + offs.value) * atten
    return view, {
        "raw_t": raw_t, "volts": volts, "present": present, "sat": sat,
        "saturated": saturated, "missing_fraction": missing_fraction,
        "trigger_s": trig.value, "deskew_s": deskew.value,
        "atten": atten, "gain_v_per_code": gain.value, "t_unit": t_unit,
        "uniform": bool(tb.get("uniform")),
    }


def _full_scale_codes(seg: Dict[str, Any]) -> float:
    fsr = seg.get("full_scale_codes")
    if isinstance(fsr, (int, float)):
        return float(fsr)
    codes = [p for p in seg.get("points", []) if p is not None]
    return float(max((abs(p) for p in codes), default=1.0)) or 1.0


def _valid_nodes(cal: Dict[str, Any]) -> np.ndarray:
    # valid interpolation nodes: present AND not saturated
    return cal["present"] & (~cal["sat"])


def _resample_to_grid(cal: Dict[str, Any], grid: np.ndarray) -> np.ndarray:
    """Linear interpolation on physical seconds.

    R2: gaps (missing points) break the interpolation chain — no reconstructed
    samples across a gap. R3: saturated points are not used as interpolation
    nodes; the saturated plateau is dropped from the resampled waveform.
    """
    raw_t, volts = cal["raw_t"], cal["volts"]
    valid = _valid_nodes(cal)
    out = np.full_like(grid, np.nan, dtype=float)
    if np.count_nonzero(valid) < 2:
        return out
    vt, vv = raw_t[valid], volts[valid]
    order = np.argsort(vt)
    vt, vv = vt[order], vv[order]
    idx = np.searchsorted(vt, grid)
    for k, g in enumerate(grid):
        j = idx[k]
        if j == 0 or j == len(vt):
            continue
        t0, t1 = vt[j - 1], vt[j]
        # gap check in original timeline: must be adjacent valid samples with
        # no missing/saturated sample strictly between them
        if _has_invalid_between(cal, t0, t1):
            continue
        frac = (g - t0) / (t1 - t0)
        out[k] = vv[j - 1] + frac * (vv[j] - vv[j - 1])
    return out


def _has_invalid_between(cal: Dict[str, Any], t0: float, t1: float) -> bool:
    raw_t, present, sat = cal["raw_t"], cal["present"], cal["sat"]
    mask = (raw_t > t0) & (raw_t < t1)
    return bool(np.any(mask & (~present | sat)))


def _score_lags(ref_grid: np.ndarray, ref_w: np.ndarray,
                grid: np.ndarray, w: np.ndarray,
                max_lag_steps: int) -> np.ndarray:
    scores = np.full(2 * max_lag_steps + 1, np.nan)
    ref_finite = np.isfinite(ref_w)
    norm_ref = np.linalg.norm(ref_w[ref_finite])
    if norm_ref == 0:
        return scores
    ref_centered = ref_w - np.mean(ref_w[ref_finite])
    for k, lag in enumerate(range(-max_lag_steps, max_lag_steps + 1)):
        shifted = np.full_like(w, np.nan)
        if lag >= 0:
            shifted[lag:] = w[:len(w) - lag]
        else:
            shifted[:lag] = w[-lag:]
        pair = ref_finite & np.isfinite(shifted)
        if np.count_nonzero(pair) < MIN_OVERLAP_FRACTION * np.count_nonzero(ref_finite):
            continue
        a = ref_centered[pair]
        b = shifted[pair] - np.mean(shifted[pair])
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            continue
        scores[k] = float(np.dot(a, b) / denom)
    return scores


def _find_peaks(scores: np.ndarray, grid_step: float) -> List[Tuple[int, float]]:
    finite = np.isfinite(scores)
    if not np.any(finite):
        return []
    best = np.nanmax(scores)
    threshold = max(0.1 * best, 0.2)
    peaks: List[Tuple[int, float]] = []
    min_sep = max(1, int(round(MIN_PEAK_SEPARATION_S / grid_step)))
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    for idx in order:
        if not finite[idx] or scores[idx] < threshold:
            continue
        if any(abs(idx - p[0]) < min_sep for p in peaks):
            continue
        peaks.append((int(idx), float(scores[idx])))
    peaks.sort(key=lambda p: -p[1])
    return peaks


def _align_segment(cal: Dict[str, Any], ref_grid: np.ndarray, ref_w: np.ndarray,
                   grid: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    step = fnum(grid[1] - grid[0])
    max_lag_steps = int(len(grid) * JITTER_PAD_FRACTION)
    w = _resample_to_grid(cal, grid)
    scores = _score_lags(ref_grid, ref_w, grid, w, max_lag_steps)
    raw_peaks = _find_peaks(scores, grid[1] - grid[0])
    global SECOND_PEAK_RATIO
    ratio = params.get("second_peak_ratio", SECOND_PEAK_RATIO)
    peaks = raw_peaks
    if len(peaks) >= 2 and peaks[1][1] >= ratio * peaks[0][1]:
        decision = "ambiguous"
    else:
        decision = "auto"
    cands = []
    for rank, (idx, score) in enumerate(peaks):
        lag_steps = idx - max_lag_steps
        lag_s = fnum(lag_steps * step)
        key = f"lag@{lag_s:.9f}s"
        cands.append({
            "key": key,
            "lag_s": lag_s,
            "lag_ns": fnum(lag_s * 1e9),
            "score": fnum(score),
            "rank": rank + 1,
            "legal": True,
        })
    return {
        "candidates": cands,
        "decision": decision,
        "grid_step_s": step,
        "resampled": w,
    }


def _measurement(view: SegmentView, cal: Dict[str, Any],
                 window: Dict[str, Any]) -> Dict[str, Any]:
    wdata = window["data"] if "data" in window else window
    wstart = convert_seconds(wdata["start"])
    wend = convert_seconds(wdata["end"])
    raw_t, volts = cal["raw_t"], cal["volts"]
    valid = _valid_nodes(cal)
    inwin = (raw_t >= cal["trigger_s"] + wstart.value) & (raw_t <= cal["trigger_s"] + wend.value)
    total = int(np.count_nonzero(inwin))
    good = inwin & valid
    n_bad = int(np.count_nonzero(inwin & ~valid))
    frac_bad = fnum(n_bad / total) if total else 1.0
    res = {
        "window_id": window.get("key", wdata.get("window_id")),
        "window_version": window.get("version", wdata.get("version")),
        "start": wstart.value,
        "end": wend.value,
        "unit": "V",
        "valid": False,
    }
    if total == 0:
        res.update({"diagnostic": "window contains no samples", "invalid_fraction": 1.0})
        view.judgments.append(_jdg(view.seg_id, "R2", "rejected",
                                   RULE_STATEMENTS["R2"],
                                   [hsid("meas", f"{view.seg_id}:{res['window_id']}:v{res['window_version']}")],
                                   {"window_id": res["window_id"], "reason": "empty window"}))
        return res
    if frac_bad > INVALID_FRACTION_LIMIT:
        res.update({"diagnostic": "缺样/饱和占比 %.0f%% 超阈，窗口不进入测量" % (100 * frac_bad),
                    "invalid_fraction": frac_bad})
        code = "R3" if cal["saturated"] and np.any(inwin & cal["sat"]) else "R2"
        view.judgments.append(_jdg(view.seg_id, code, "rejected",
                                   RULE_STATEMENTS[code],
                                   [hsid("meas", f"{view.seg_id}:{res['window_id']}:v{res['window_version']}")],
                                   {"window_id": res["window_id"],
                                    "invalid_fraction": frac_bad}))
        return res
    vals = volts[good & inwin]
    vpp = fnum(float(np.max(vals) - np.min(vals)))
    mean = fnum(float(np.mean(vals)))
    res.update({"valid": True, "vpp": vpp, "mean": mean, "invalid_fraction": frac_bad})
    if cal["saturated"] and np.any(inwin & cal["sat"]):
        res["note"] = "窗口含饱和样本，已从测量中剔除"
        view.judgments.append(_jdg(view.seg_id, "R3", "accepted",
                                   RULE_STATEMENTS["R3"],
                                   [hsid("meas", f"{view.seg_id}:{res['window_id']}:v{res['window_version']}")],
                                   {"window_id": res["window_id"],
                                    "dropped_saturated": int(np.count_nonzero(inwin & cal["sat"]))}))
    return res


def _at_boundary(raw_t: np.ndarray, trigger_s: float) -> bool:
    span = raw_t[-1] - raw_t[0]
    pad = JITTER_PAD_FRACTION * span
    return bool(trigger_s <= raw_t[0] + pad or trigger_s >= raw_t[-1] - pad)




def _rule_token(params: Dict[str, Any]) -> str:
    keys = sorted(params.keys())
    return "params(" + ",".join(f"{k}={params[k]}" for k in keys) + ")"


def derive_state(materials: List[Dict[str, Any]],
                 params: Dict[str, Any],
                 rulings: List[Dict[str, Any]],
                 windows: List[Dict[str, Any]],
                 event_id: str) -> DerivedState:
    segment_mats = [m for m in materials if m["grp"] == "segment"]
    segment_mats.sort(key=lambda m: (m["ord"], m["key"]))
    items: Dict[str, Dict[str, Any]] = {}
    segments: List[SegmentView] = []
    ids: Dict[str, Dict[str, str]] = {}
    selected: Dict[str, str] = {}
    mat_of = {m["data"]["segment_id"]: m for m in segment_mats}

    def add_item(kind: str, sid: str, token: str, deps: List[str],
                 obj: Dict[str, Any]) -> str:
        rid = hsid(kind, token)
        items[rid] = {"id": rid, "kind": kind, "segment_id": sid,
                      "deps": deps, "obj": obj}
        return rid

    # 1) calibration (timebase/vertical/probe/deskew) per segment
    cals: Dict[str, Dict[str, Any]] = {}
    ref_view: Optional[SegmentView] = None
    for mat in segment_mats:
        view, cal = _calibrate_segment(mat)
        segments.append(view)
        ids[view.seg_id] = {}
        if cal is None:
            continue
        cals[view.seg_id] = cal
        seg_id = view.seg_id
        valid = _valid_nodes(cal)
        seg_obj = {
            "segment_id": seg_id, "channel": view.channel,
            "material_id": mat["id"], "uniform": cal["uniform"],
            "point_count": int(len(cal["raw_t"])),
            "missing_fraction": cal["missing_fraction"],
            "saturated": cal["saturated"],
            "trigger_rep": view.original.get("trigger"),
            "trigger_s": fnum(cal["trigger_s"]),
            "deskew_rep": view.original.get("deskew"),
            "deskew_s": fnum(cal["deskew_s"]),
            "gain_rep": view.original.get("vertical_gain"),
            "probe_attenuation": cal["atten"],
            "unit_basis": {"time": "s", "voltage": "V@probe_tip"},
        }
        ids[seg_id]["seg"] = add_item(
            "seg", seg_id, f"segment:{mat['id']}", [mat["id"]], seg_obj)
        ta = {"segment_id": seg_id, "uniform": cal["uniform"],
              "start_s": fnum(float(cal["raw_t"][0])),
              "times_s": [fnum(float(x)) for x in cal["raw_t"]],
              "original_rep": cal["t_unit"]}
        ids[seg_id]["ta"] = add_item(
            "ta", seg_id, f"timeaxis:{mat['id']}", [ids[seg_id]["seg"]], ta)
        wf = {"segment_id": seg_id,
              "volts": [None if not np.isfinite(v) else fnum(float(v))
                        for v in cal["volts"]],
              "valid_mask": [bool(x) for x in valid],
              "saturated_mask": [bool(x) for x in cal["sat"]],
              "unit": "V"}
        ids[seg_id]["wf"] = add_item(
            "wf", seg_id, f"waveform:{mat['id']}",
            [ids[seg_id]["seg"], ids[seg_id]["ta"]], wf)
        if view.reference:
            ref_view = view

    if ref_view is None:
        raise ValueError("reference segment missing")
    ref_cal = cals[ref_view.seg_id]
    rstep = float(np.median(np.diff(ref_cal["raw_t"])))
    ref_grid = ref_cal["raw_t"][0] + rstep * np.arange(GRID_POINTS)
    ref_w = _resample_to_grid(ref_cal, ref_grid)

    def ruling_for(seg_id: str):
        acc = [r for r in rulings if r.get("seg_id") == seg_id]
        accept = [k for r in acc for k in r.get("accept", [])]
        reject = [k for r in acc for k in r.get("reject", [])]
        return acc, accept, reject

    # 2) multi-peak alignment + rulings
    jitter_rows: List[Tuple[SegmentView, float, str]] = []
    for view in segments:
        seg_id = view.seg_id
        cal = cals.get(seg_id)
        if cal is None:
            continue
        acc_events, accept, reject = ruling_for(seg_id)
        if view.reference:
            cand = {"key": "lag@0.000000000s", "lag_s": 0.0, "lag_ns": 0.0,
                    "score": 1.0, "rank": 1, "legal": True, "status": "selected"}
            al = {"segment_id": seg_id, "candidates": [cand],
                  "decision": "reference", "status": "selected",
                  "selected": cand["key"], "selected_lag_ns": 0.0}
            tok = f"align:{seg_id}:reference"
            ids[seg_id]["align"] = add_item(
                "align", seg_id, tok, [ids[seg_id]["wf"]], al)
            view.align = al
            selected[seg_id] = cand["key"]
            jitter_rows.append((view, cal["trigger_s"] + cal["deskew_s"], "reference"))
            continue

        a = _align_segment(cal, ref_grid, ref_w, ref_grid, params)
        legal = [c for c in a["candidates"] if c["key"] not in reject]
        chosen = None
        if accept:
            chosen = next((c for c in legal if c["key"] == accept[-1]), None)
        if chosen is None and len(legal) == 1:
            chosen = legal[0]
        status = "selected" if chosen is not None else "pending"
        for c in a["candidates"]:
            c["status"] = ("rejected" if c["key"] in reject
                           else "selected" if chosen is not None and c is chosen
                           else "candidate")
        al = {"segment_id": seg_id, "candidates": a["candidates"],
              "decision": a["decision"], "status": status,
              "selected": chosen["key"] if chosen else None,
              "selected_lag_ns": chosen["lag_ns"] if chosen else None}
        tok = (f"align:{seg_id}:{_rule_token(params)}:"
               + "|".join(r["id"] for r in acc_events))
        ids[seg_id]["align"] = add_item(
            "align", seg_id, tok,
            [ids[seg_id]["wf"], _rule_token(params)], al)
        view.align = al
        if a["decision"] == "ambiguous":
            view.judgments.append(_jdg(
                seg_id, "R4", "accepted" if chosen else "diagnostic",
                RULE_STATEMENTS["R4"], [ids[seg_id]["align"]],
                {"candidate_count": len(a["candidates"]),
                 "scores": [c["score"] for c in a["candidates"]],
                 "resolution": "ruling" if chosen else "pending"}))
        if cal["deskew_s"] != 0.0:
            view.judgments.append(_jdg(
                seg_id, "R5", "accepted", RULE_STATEMENTS["R5"],
                [ids[seg_id]["align"]],
                {"deskew_ns": fnum(cal["deskew_s"] * 1e9)}))
        if chosen is not None:
            selected[seg_id] = chosen["key"]
            corr_trig = cal["trigger_s"] + cal["deskew_s"] - chosen["lag_s"]
            if _at_boundary(cal["raw_t"], cal["trigger_s"]):
                view.judgments.append(_jdg(
                    seg_id, "R6", "rejected", RULE_STATEMENTS["R6"],
                    [ids[seg_id].get("jitter", "jitter:excluded")],
                    {"trigger_s": fnum(cal["trigger_s"]),
                     "record_start_s": fnum(float(cal["raw_t"][0])),
                     "record_end_s": fnum(float(cal["raw_t"][-1]))}))
            else:
                jid = add_item(
                    "jitter", seg_id, f"jitter:{seg_id}", [ids[seg_id]["align"]],
                    {"segment_id": seg_id, "status": status, "unit": "ns"})
                ids[seg_id]["jitter"] = jid
                jitter_rows.append((view, corr_trig, status))

    # 3) jitter distribution (R6 keeps boundary triggers out)
    jdist_deps: List[str] = []
    if jitter_rows:
        median_trig = float(np.median([t for _, t, _ in jitter_rows]))
        for view, corr_trig, status in jitter_rows:
            dev_ns = fnum((corr_trig - median_trig) * 1e9)
            view.jitter_ns = dev_ns
            items[ids[view.seg_id]["jitter"]]["obj"].update(
                {"corrected_trigger_s": fnum(corr_trig),
                 "deviation_ns": dev_ns})
            jdist_deps.append(ids[view.seg_id]["jitter"])
        jit = {"n": len(jitter_rows),
               "median_trigger_s": fnum(median_trig),
               "rms_ns": fnum(float(np.sqrt(np.mean(
                   [(t - median_trig) ** 2 for _, t, _ in jitter_rows]))) * 1e9),
               "excluded_boundary_segments": [
                   v.seg_id for v in segments
                   if any(j["code"] == "R6" for j in v.judgments)],
               "unit": "ns"}
        add_item("jdist", "all", f"jdist:{event_id}", jdist_deps, jit)

    # 4) measurement windows (versioned; R2/R3 reject bad windows; R7 provenance)
    for view in segments:
        seg_id = view.seg_id
        cal = cals.get(seg_id)
        if cal is None:
            continue
        affected: List[str] = []
        for win in windows:
            m = _measurement(view, cal, win)
            tok = (f"meas:{seg_id}:{win['key']}:v{win['version']}:"
                   f"{mat_of[seg_id]['id']}")
            mid = add_item("meas", f"{seg_id}:{win['key']}", tok,
                           [ids[seg_id]["seg"], win["id"]], m)
            affected.append(mid)
            view.measurements.append(m)
        view.judgments.append(_jdg(
            seg_id, "R7", "accepted", RULE_STATEMENTS["R7"], affected,
            {"window_versions": {w["key"]: w["version"] for w in windows}}))

    all_judgments: List[Dict[str, Any]] = []
    for view in segments:
        all_judgments.extend(view.judgments)
        for er in view.excluded_reasons:
            all_judgments.append({
                "id": er["judgment_id"], "segment_id": view.seg_id,
                "code": er["code"], "rules_version": er["rules_version"],
                "status": "rejected", "basis": er["reason"],
                "affected_ids": [], "evidence": {"diagnostic": er["diagnostic"]},
            })
    all_judgments = _dedupe(all_judgments)
    for view in segments:
        view.judgments = [j for j in all_judgments if j["segment_id"] == view.seg_id]

    return DerivedState(
        items=items,
        segments=segments,
        judgments=all_judgments,
        params=params,
        windows=windows,
        segment_order=[v.seg_id for v in segments],
        selected_align=selected,
        rulings=rulings,
    )


def invalidate(old: DerivedState, new: DerivedState) -> Dict[str, Any]:
    """Diff two derived states: retained / changed ids and forward impact."""
    old_sig = {i["id"]: _item_signature(i) for i in old.items.values()}
    new_sig = {i["id"]: _item_signature(i) for i in new.items.values()}
    retained = sorted(set(old_sig) & set(new_sig)
                      and [k for k in old_sig if k in new_sig and old_sig[k] == new_sig[k]])
    changed = sorted({k for k in set(old_sig) | set(new_sig)
                      if old_sig.get(k) != new_sig.get(k)})
    return {
        "retained_ids": retained,
        "changed_ids": changed,
        "added_ids": sorted(set(new_sig) - set(old_sig)),
        "removed_ids": sorted(set(old_sig) - set(new_sig)),
    }


def _item_signature(item: Dict[str, Any]) -> str:
    import json
    return hashlib.sha256(
        json.dumps(item["obj"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
