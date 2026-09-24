"""派生层（规范化发生在此处）：

- J1 时间轴不等间隔时不用索引相减（物理时间口径）
- J2 缺样与饱和区间不进入插值/打分
- J3 对齐保留多峰互相关候选，不唯一时保留全部合法候选
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from station import repo
from station.ids import candidate_id
from station.units import parse_probe, parse_quantity

REFERENCE_SEGMENT = 1
MIN_OVERLAP_DEFAULT = 0.6
PROMINENCE_DEFAULT = 0.90


@dataclass
class SegmentState:
    no: int
    label: str
    channel: str
    note: str
    materials: dict
    times_ns: np.ndarray | None
    samples: np.ndarray | None
    missing_mask: np.ndarray | None
    saturated_mask: np.ndarray | None
    usable_mask: np.ndarray | None
    valid_mask: np.ndarray | None
    gain_v_per_adc: float | None
    probe_ratio: float | None
    trigger_ns: float | None
    deskew_ns: float | None
    window: dict | None
    diagnostics: list
    excluded: list
    rail_high: float | None
    rail_low: float | None


def _material_payload(active: dict, no: int, kind: str) -> dict | None:
    return active["materials"].get((no, kind))


def _build_segment(active: dict, no: int) -> SegmentState:
    meta = active["segment_meta"][no]
    mats = {k: _material_payload(active, no, k) for k in repo.MATERIAL_KINDS}
    diagnostics: list[str] = []

    tb = mats["timebase"]
    if tb.get("kind") == "irregular" or (
        tb.get("extra") and tb["extra"].get("kind") == "irregular"
    ):
        raw_times = tb["raw_repr"]
        times = np.array([np.nan if t is None else float(t) for t in raw_times])
        irregular = True
    else:
        n = len(mats["samples"]["raw_repr"])
        dt_q = parse_quantity(tb["raw_repr"], tb["unit"], tb["coordinate_basis"], "time")
        diagnostics += dt_q.diagnostics
        dt = dt_q.value
        if dt is None:
            times = None
        else:
            times = np.arange(n, dtype=float) * dt
        irregular = dt is not None and bool(
            (tb.get("extra") or {}).get("kind") == "irregular"
        )

    sp = mats["samples"]
    raw = sp["raw_repr"]
    samples = np.array([np.nan if v is None else float(v) for v in raw])
    declared_missing = set((sp.get("extra") or {}).get("missing_indices", []))
    rail_high = (sp.get("extra") or {}).get("saturation", {}).get("rail_high")
    rail_low = (sp.get("extra") or {}).get("saturation", {}).get("rail_low", 0)
    missing_mask = np.array(
        [i in declared_missing or np.isnan(v) for i, v in enumerate(samples)]
    )
    saturated_mask = np.array(
        [(not np.isnan(v)) and (
            (rail_high is not None and v >= float(rail_high) - 1e-9) or
            (rail_low is not None and v <= float(rail_low) + 1e-9))
         for v in samples]
    )
    usable_mask = ~(missing_mask | saturated_mask)
    valid_mask = ~missing_mask

    gain_q = parse_quantity(mats["gain"]["raw_repr"], mats["gain"]["unit"],
                            mats["gain"]["coordinate_basis"], "gain")
    diagnostics += [f"S{no} gain: {d}" for d in gain_q.diagnostics]

    probe_q = parse_probe(mats["probe"]["raw_repr"])
    diagnostics += [f"S{no} probe: {d}" for d in probe_q.diagnostics]

    trig_q = parse_quantity(mats["trigger"]["raw_repr"], mats["trigger"]["unit"],
                            mats["trigger"]["coordinate_basis"], "time")
    diagnostics += [f"S{no} trigger: {d}" for d in trig_q.diagnostics]

    dk_q = parse_quantity(mats["deskew"]["raw_repr"], mats["deskew"]["unit"],
                          mats["deskew"]["coordinate_basis"], "time")
    diagnostics += [f"S{no} deskew: {d}" for d in dk_q.diagnostics]

    win_mat = mats["window"]
    ws = parse_quantity(win_mat["raw_repr"] if False else
                        win_mat["payload_window_start"], "ns",
                        "relative_to_segment_origin", "time") \
        if False else None
    w_start = parse_quantity(
        (win_mat.get("extra") or {}).get("window_start",
                                         win_mat.get("raw_repr")),
        "ns", "relative_to_segment_origin", "time") \
        if False else None
    # window material payload 在导入时展开为 start/end（见 _window_fields）
    wextra = win_mat.get("extra") or {}
    ws_q = parse_quantity(wextra.get("start_raw"),
                          wextra.get("start_unit", "ns"),
                          "relative_to_segment_origin", "time") \
        if wextra.get("start_raw") is not None else None
    we_q = parse_quantity(wextra.get("end_raw"),
                          wextra.get("end_unit", "ns"),
                          "relative_to_segment_origin", "time") \
        if wextra.get("end_raw") is not None else None
    window = None
    if ws_q is not None and we_q is not None:
        diagnostics += [f"S{no} window: {d}" for d in ws_q.diagnostics + we_q.diagnostics]
        window = {"version": win_mat.get("version"),
                  "start_ns": ws_q.value, "end_ns": we_q.value,
                  "start_diag": ws_q.diagnostics, "end_diag": we_q.diagnostics}

    excluded: list[dict] = []
    if times is None:
        excluded.append({"code": "TIMEBASE_UNPARSEABLE",
                         "reason": "时基无法转换，不能按物理时间对齐"})
    if usable_mask.sum() < 8:
        excluded.append({"code": "INSUFFICIENT_USABLE_SAMPLES",
                         "reason": "缺样/饱和后可用采样不足"})

    return SegmentState(
        no=no, label=meta.get("label", f"S{no}"), channel=meta.get("channel"),
        note=meta.get("note", ""), materials=mats, times_ns=times, samples=samples,
        missing_mask=missing_mask, saturated_mask=saturated_mask,
        usable_mask=usable_mask, valid_mask=valid_mask,
        gain_v_per_adc=gain_q.value, probe_ratio=probe_q.value,
        trigger_ns=trig_q.value, deskew_ns=dk_q.value, window=window,
        diagnostics=diagnostics, excluded=excluded,
        rail_high=float(rail_high) if rail_high is not None else None,
        rail_low=float(rail_low) if rail_low is not None else None,
    )


def _nearest_value(times, values, usable, t):
    """最近可用采样；缺样与饱和点不作为插值/取值来源（J2）。"""
    idx = int(np.argmin(np.abs(times - t)))
    if usable[idx]:
        return values[idx], idx
    order = np.argsort(np.abs(times - t))
    for j in order:
        if usable[j]:
            return values[j], int(j)
    return np.nan, None


def align_segment(ref: SegmentState, seg: SegmentState, params: dict) -> dict:
    """以物理时间为口径的互相关；返回全部候选与排除判定。"""
    min_overlap = float(params.get("min_overlap", MIN_OVERLAP_DEFAULT))
    excluded = list(seg.excluded)
    if ref.excluded:
        excluded.append({"code": "REFERENCE_EXCLUDED",
                         "reason": f"参考段 S{ref.no} 已被排除"})

    ref_t, ref_v = ref.times_ns, ref.samples
    t = seg.times_ns
    deskew = seg.deskew_ns if seg.deskew_ns is not None else 0.0
    t_shift = t - deskew  # CH 标定 deskew 先在物理时间轴上扣除

    lags = np.round(np.arange(-30, 30.0001, 1.0), 6)
    span = float(min(ref_t[-1], t_shift[-1]) - max(ref_t[0], t_shift[0]))
    total_span = float(max(ref_t[-1], t_shift[-1]) - min(ref_t[0], t_shift[0]))
    candidates = []
    boundary = False
    for lag in lags:
        shifted = t_shift + lag
        lo = max(ref_t[0], shifted[0])
        hi = min(ref_t[-1], shifted[-1])
        if hi <= lo or total_span == 0 or (hi - lo) / total_span < min_overlap:
            continue
        weight_sum = 0.0
        err_sum = 0.0
        count = 0
        for rt, rv, ru in zip(ref_t, ref_v, ref.usable_mask):
            if not ru:
                continue
            if rt < lo or rt > hi:
                continue
            j = int(np.argmin(np.abs(shifted - rt)))
            if not seg.usable_mask[j]:
                continue  # J2：缺样/饱和点不进入打分
            err_sum += (rv - seg.samples[j]) ** 2
            weight_sum += 1.0
            count += 1
        if count < 6:
            continue
        mse = err_sum / weight_sum
        score = 1.0 / (1.0 + mse / 400.0)
        cid = candidate_id(f"S{seg.no}", float(lag), float(score),
                           params.get("rule_version", "derive-1.0.0"))
        candidates.append({"candidate_id": cid, "lag_ns": float(lag),
                           "score": round(float(score), 6), "overlap_ns": round(float(hi - lo), 6)})
    if not candidates:
        excluded.append({"code": "NO_ALIGNMENT_CANDIDATE",
                         "reason": "重叠区内可用样本不足，没有合法互相关候选"})
    best = max(c["score"] for c in candidates) if candidates else None
    tol = 2e-4
    top = [c for c in candidates if abs(c["score"] - best) <= tol]
    unresolved = len(top) > 1
    # 触发在边界：最优候选落在搜索范围端点，或事件贴近采集起点
    if top and any(abs(c["lag_ns"] - (-30.0)) < 1e-9 or
                   abs(c["lag_ns"] - 30.0) < 1e-9 for c in top):
        boundary = True
    if seg.times_ns is not None and seg.samples is not None:
        usable_t = seg.times_ns[seg.usable_mask]
        usable_v = seg.samples[seg.usable_mask]
        pk_idx = int(np.argmax(usable_v))
        if usable_t[pk_idx] <= 3.0:
            boundary = True
    chosen = top[0] if len(top) == 1 else None
    return {"subject": f"S{seg.no}", "channel": seg.channel,
            "candidates": candidates, "top_candidates": top,
            "chosen": chosen, "unresolved": unresolved,
            "deskew_ns": deskew, "boundary_trigger": boundary,
            "excluded": excluded}


def measure_segment(seg: SegmentState, align: dict) -> dict:
    win = seg.window
    result: dict[str, Any] = {"subject": f"S{seg.no}", "available": False}
    if align.get("excluded") and not align.get("candidates"):
        result["reason"] = "对齐无合法候选"
        return result
    if win is None or win["start_ns"] is None or win["end_ns"] is None:
        result["reason"] = "测量区间无法转换"
        return result
    chosen = align.get("chosen")
    if chosen is None:
        result["reason"] = "对齐不唯一，测量挂起等待人工裁定"
        result["candidate_lags_ns"] = [c["lag_ns"] for c in align["top_candidates"]]
        return result
    lag = chosen["lag_ns"]
    deskew = align.get("deskew_ns", 0.0)
    lo = win["start_ns"] - lag + deskew
    hi = win["end_ns"] - lag + deskew
    in_win = (seg.times_ns >= lo) & (seg.times_ns <= hi) & seg.usable_mask
    if in_win.sum() < 2:
        result["reason"] = "测量窗内可用样本不足（缺样/饱和已排除）"
        return result
    peak_adc = float(np.max(seg.samples[in_win]))
    peak_time_ns = float(seg.times_ns[in_win][np.argmax(seg.samples[in_win])])
    result.update({
        "available": True, "window_version": win["version"],
        "window_start_ns": win["start_ns"], "window_end_ns": win["end_ns"],
        "aligned_window_ns": [round(lo, 6), round(hi, 6)],
        "peak_adc": round(peak_adc, 6), "peak_time_ns": round(peak_time_ns, 6),
        "peak_voltage": (round((peak_adc - 128.0) * seg.gain_v_per_adc
                               * (seg.probe_ratio or 1.0), 6)
                         if seg.gain_v_per_adc is not None else None),
        "voltage_diagnostic": (
            None if seg.gain_v_per_adc is not None
            else "竖直增益无法转换，电压不可用，仅保留 ADC"),
        "n_used": int(in_win.sum()),
        "n_missing_in_window": int(((seg.times_ns >= lo) & (seg.times_ns <= hi)
                                    & seg.missing_mask).sum()),
        "n_saturated_in_window": int(((seg.times_ns >= lo) & (seg.times_ns <= hi)
                                      & seg.saturated_mask).sum()),
    })
    return result


# 测量区间 material 的展开由 repo.window_fields / repo.material_record_from_window 负责
material_record_from_window = repo.material_record_from_window


def load_active_state(conn, project_id: str, branch_id: str,
                      as_of_event_id: str | None = None) -> dict:
    """沿事件链折叠到某事件（默认分支头），恢复当时的材料与参数版本。"""
    events = [dict(r) for r in conn.execute(
        "SELECT * FROM events WHERE branch_id=? ORDER BY branch_seq", (branch_id,))]
    from station.db import jloads

    folded: list[dict] = []
    for ev in events:
        folded.append(ev)
        if as_of_event_id is not None and ev["event_id"] == as_of_event_id:
            break
        if as_of_event_id is None:
            continue

    active_materials: dict[tuple[int, str], dict] = {}
    material_meta: dict[int, dict] = {}
    rule_versions: dict = {"derive": "derive-1.0.0",
                           "peak_prominence_ratio": PROMINENCE_DEFAULT,
                           "min_overlap": MIN_OVERLAP_DEFAULT}
    decisions: dict[str, dict] = {}
    undo_until: set[str] = set()
    started = False
    last_event = None
    for ev in folded:
        last_event = ev
        payload = jloads(ev["payload"])
        if ev["type"] == "import_batch":
            started = True
            for row in conn.execute(
                "SELECT * FROM materials WHERE project_id=? ORDER BY segment_no, kind",
                (project_id,)):
                mp = jloads(row["payload"])
                active_materials[(mp["segment_no"], mp["kind"])] = mp
                material_meta.setdefault(mp["segment_no"],
                                         {"label": mp.get("label"),
                                          "channel": mp.get("channel")})
            rule_versions.update(payload.get("rule_versions", {}))
        elif ev["type"] == "material_revised":
            no = payload["segment_no"]
            kind = payload["kind"]
            old_id = active_materials.get((no, kind), {}).get("material_id")
            new_payload = payload["new_material"]
            active_materials[(no, kind)] = new_payload
        elif ev["type"] == "window_revised":
            no = payload["segment_no"]
            active_materials[(no, "window")] = payload["new_material"]
        elif ev["type"] == "rules_revised":
            rule_versions.update(payload["changes"])
            rule_versions["rule_set_id"] = payload.get("rule_set_id")
        elif ev["type"] == "decision":
            key = f"S{payload['segment_no']}:{payload['topic']}"
            decisions[key] = {"decision": payload["decision"],
                              "candidate_id": payload.get("candidate_id"),
                              "lag_ns": payload.get("lag_ns"),
                              "event_id": ev["event_id"],
                              "seq": ev["branch_seq"],
                              "undone": False,
                              "detail": payload.get("detail")}
        elif ev["type"] == "decision_undone":
            key = f"S{payload['segment_no']}:{payload['topic']}"
            if key in decisions:
                decisions[key]["undone"] = True
    # 分叉事件（branch_forked）不改本分支状态
    return {
        "project_id": project_id,
        "branch_id": branch_id,
        "head_event_id": last_event["event_id"] if last_event else None,
        "head_seq": last_event["branch_seq"] if last_event else 0,
        "materials": active_materials,
        "segment_meta": material_meta,
        "rule_versions": rule_versions,
        "decisions": decisions,
    }


def _material_id_map(active: dict) -> dict[tuple[int, str], str]:
    from station.ids import material_id

    return {key: material_id(mp) for key, mp in active["materials"].items()}


def derive_state(conn, project_id: str, branch_id: str,
                 as_of_event_id: str | None = None, persist: bool = True) -> dict:
    active = load_active_state(conn, project_id, branch_id, as_of_event_id)
    nos = sorted(active["segment_meta"])
    segments: dict[int, SegmentState] = {}
    for no in nos:
        segments[no] = _build_segment(active, no)

    ref = segments.get(REFERENCE_SEGMENT)
    params = {
        "rule_version": active["rule_versions"].get("rule_set_id",
                                                    active["rule_versions"]["derive"]),
        "peak_prominence_ratio": active["rule_versions"].get(
            "peak_prominence_ratio", PROMINENCE_DEFAULT),
        "min_overlap": active["rule_versions"].get(
            "min_overlap", MIN_OVERLAP_DEFAULT),
    }
    idmap = _material_id_map(active)

    alignments: dict[int, dict] = {}
    measurements: dict[int, dict] = {}
    judgments: list[dict] = []
    for no in nos:
        seg = segments[no]
        irregular = bool(seg.times_ns is not None and
                         np.any(np.abs(np.diff(seg.times_ns)[
                             ~np.isnan(np.diff(seg.times_ns))] -
                             np.nanmedian(np.diff(seg.times_ns)))[
                             ~np.isnan(np.abs(np.diff(seg.times_ns)[
                                 ~np.isnan(np.diff(seg.times_ns))] -
                                 np.nanmedian(np.diff(seg.times_ns))))] > 1e-9))
        # 用时基 material 的 irregular 标志更直接
        tb_mp = active["materials"].get((no, "timebase"))
        irregular = (tb_mp or {}).get("extra", {}).get("kind") == "irregular"
        if irregular:
            judgments.append({
                "code": "J1_PHYSICAL_TIME",
                "subject": f"S{no}",
                "detail": {"rule": "时间轴不等间隔时不用索引相减",
                           "basis": "sample_times_relative_to_segment_origin",
                           "action": "对齐与抖动一律使用物理时间(ns)，不使用索引差"},
            })
        if seg.missing_mask.any():
            judgments.append({
                "code": "J2_NO_INTERPOLATION",
                "subject": f"S{no}",
                "detail": {"rule": "缺样区间不进入插值",
                           "missing_indices": [int(i) for i in
                                               np.where(seg.missing_mask)[0]]},
            })
        if seg.saturated_mask.any():
            judgments.append({
                "code": "J2_NO_INTERPOLATION",
                "subject": f"S{no}",
                "detail": {"rule": "饱和区间不进入插值",
                           "saturated_indices": [int(i) for i in
                                                 np.where(seg.saturated_mask)[0]],
                           "rail_high": seg.rail_high},
            })
        if no == REFERENCE_SEGMENT:
            alignments[no] = {"subject": f"S{no}", "channel": seg.channel,
                              "candidates": [], "top_candidates": [],
                              "chosen": {"lag_ns": 0.0, "score": 1.0,
                                         "candidate_id": "cand_reference"},
                              "unresolved": False, "deskew_ns": seg.deskew_ns or 0.0,
                              "boundary_trigger": False, "excluded": []}
            measurements[no] = measure_segment(seg, alignments[no])
            continue
        align = align_segment(ref, seg, params)
        if align["unresolved"]:
            judgments.append({
                "code": "J3_MULTI_PEAK_CANDIDATES",
                "subject": f"S{no}",
                "detail": {"rule": "对齐保留多峰互相关候选",
                           "candidate_ids": [c["candidate_id"]
                                             for c in align["top_candidates"]],
                           "lags_ns": [c["lag_ns"] for c in align["top_candidates"]],
                           "scores": [c["score"] for c in align["top_candidates"]]},
            })
        key = f"S{no}:alignment"
        decision = active["decisions"].get(key)
        if decision and not decision.get("undone"):
            chosen = next((c for c in align["candidates"]
                           if c["candidate_id"] == decision["candidate_id"]), None)
            align["chosen"] = chosen
            align["unresolved"] = False
            align["resolved_by"] = decision["event_id"]
        alignments[no] = align
        measurements[no] = measure_segment(seg, align)

    # 触发抖动：物理时间口径 jitter = (触发时标差) - (事件绝对位置差)
    jitters = []
    ref_trig = ref.trigger_ns if ref else None
    for no in nos:
        seg = segments[no]
        al = alignments.get(no, {})
        if seg.trigger_ns is None or ref_trig is None or al.get("chosen") is None:
            jitters.append({"segment_no": no, "jitter_ns": None,
                            "reason": "触发时标或对齐不可用"})
            continue
        peak_idx = int(np.argmax(seg.samples[seg.usable_mask]))
        event_local = float(seg.times_ns[seg.usable_mask][peak_idx]) - \
            (al.get("deskew_ns", 0.0) or 0.0) - al["chosen"]["lag_ns"]
        ref_event = 12.0
        jitter = (seg.trigger_ns - ref_trig) - (event_local - ref_event)
        jitters.append({"segment_no": no, "jitter_ns": round(float(jitter), 6),
                        "lag_ns": al["chosen"]["lag_ns"],
                        "candidate_id": al["chosen"].get("candidate_id"),
                        "boundary_trigger": bool(al.get("boundary_trigger"))})

    view = {
        "active": active,
        "segments": _segments_view(segments),
        "alignments": alignments,
        "measurements": measurements,
        "jitters": jitters,
        "judgments": judgments,
        "params": params,
        "material_ids": {f"{k[0]}:{k[1]}": v for k, v in idmap.items()},
    }
    if persist:
        _persist_view(conn, view, idmap, params)
    return view


def _persist_view(conn, view: dict, idmap: dict, params: dict) -> None:
    from station.db import jdumps

    project_id = view["active"]["project_id"]
    for no, align in view["alignments"].items():
        deps = [idmap[(no, k)] for k in
                ("samples", "timebase", "gain", "probe", "deskew", "trigger")
                if (no, k) in idmap]
        if no != REFERENCE_SEGMENT and (REFERENCE_SEGMENT, "samples") in idmap:
            deps.append(idmap[(REFERENCE_SEGMENT, "samples")])
            deps.append(idmap[(REFERENCE_SEGMENT, "timebase")])
        repo.store_derivation(
            conn, project_id, "alignment", f"S{no}", params["rule_version"], deps,
            {"chosen": align.get("chosen"),
             "top": [c["candidate_id"] for c in align.get("top_candidates", [])],
             "unresolved": align.get("unresolved"),
             "boundary": align.get("boundary_trigger")})
        win_dep = idmap.get((no, "window"))
        mdeps = list(deps) + ([win_dep] if win_dep else [])
        repo.store_derivation(
            conn, project_id, "measurement", f"S{no}", params["rule_version"], mdeps,
            view["measurements"][no])
        repo.store_derivation(
            conn, project_id, "jitter", f"S{no}", params["rule_version"],
            deps + ([idmap[(REFERENCE_SEGMENT, "trigger")]]
                    if no != REFERENCE_SEGMENT and
                    (REFERENCE_SEGMENT, "trigger") in idmap else []),
            next(j for j in view["jitters"] if j["segment_no"] == no))


def _segments_view(segments: dict[int, SegmentState]) -> dict:
    out = {}
    for no, s in segments.items():
        out[no] = {
            "segment_no": no, "label": s.label, "channel": s.channel,
            "note": s.note,
            "times_ns": [None if t is None or (isinstance(t, float) and np.isnan(t))
                         else round(float(t), 6) for t in
                         (s.times_ns if s.times_ns is not None else [])],
            "samples": [None if np.isnan(v) else int(v) for v in s.samples],
            "missing_indices": [int(i) for i in np.where(s.missing_mask)[0]],
            "saturated_indices": [int(i) for i in np.where(s.saturated_mask)[0]],
            "usable_indices": [int(i) for i in np.where(s.usable_mask)[0]],
            "gain_v_per_adc": s.gain_v_per_adc,
            "probe_ratio": s.probe_ratio, "trigger_ns": s.trigger_ns,
            "deskew_ns": s.deskew_ns, "window": s.window,
            "diagnostics": s.diagnostics, "excluded": s.excluded,
            "raw_materials": {k: v for k, v in s.materials.items()},
        }
    return out
