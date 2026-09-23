"""状态重放：从事件序列重建分支可见结论，并显式保存派生层与依赖。

- 任何旧事件都可恢复：upto_seq 截断即得当时的依赖版本与可见结论；
- undo/restore 本身也是事件，不改写历史；
- 派生层（多候选、抖动、测量）连同 deps 显式落库，供失效范围计算。
"""
from __future__ import annotations

import numpy as np

import core
import store


def replay_active(events):
    """undo 移除目标事件的效果；restore 截断到指定 seq。历史本身不变。"""
    active = []
    for e in events:
        p = e["payload"]
        if e["type"] == "undo":
            active = [a for a in active if a["seq"] != p["target_seq"]]
        elif e["type"] == "restore":
            active = [a for a in active if a["seq"] <= p["to_seq"]]
        else:
            active.append(e)
    return active


def _nan_to_none(x):
    if x is None:
        return None
    try:
        return None if np.isnan(x) else float(x)
    except TypeError:
        return float(x)


def compute_state(conn, branch_id, upto_seq=None):
    events = store.gather_events(conn, branch_id)
    if upto_seq is not None:
        events = [e for e in events if e["seq"] <= upto_seq]
    head_seq = events[-1]["seq"] if events else 0
    active = replay_active(events)

    sources = store.get_sources(conn)
    active_versions = {sid: min(e["versions"]) for sid, e in sources.items()}
    interval_version = "mi-v1"
    chosen = {}
    rejected = {}
    invalidations = []
    for e in active:
        p = e["payload"]
        if e["type"] == "param-revision":
            interval_version = p["to"]
            invalidations.append({
                "event_seq": e["seq"], "event_id": e["event_id"],
                "changed": {"params": [p["from"]]},
                "affected": p.get("affected", [])})
        elif e["type"] == "source-change":
            active_versions[p["source_id"]] = p["version"]
            invalidations.append({
                "event_seq": e["seq"], "event_id": e["event_id"],
                "changed": {"sources": [p["source_id"]]},
                "affected": p.get("affected", [])})
        elif e["type"] == "ruling":
            seg = p["segment_id"]
            if p["decision"] == "accept":
                chosen[seg] = p["candidate_id"]
            else:
                rejected.setdefault(seg, [])
                if p["candidate_id"] not in rejected[seg]:
                    rejected[seg].append(p["candidate_id"])

    def src(sid):
        return sources[sid]["versions"][active_versions[sid]]

    channels = {sid: e["versions"][active_versions[sid]]
                for sid, e in sources.items() if e["kind"] == "channel"}
    intervals = src("intervals")
    interval = next(v for v in intervals["versions"]
                    if v["version"] == interval_version)
    win_start = core.to_seconds(interval["start"])
    win_end = core.to_seconds(interval["end"])

    seg_ids = sorted(sid for sid, e in sources.items() if e["kind"] == "segment")
    norms = {}
    for sid in seg_ids:
        seg = src(sid)
        ch_deskew = core.to_seconds(channels["cal-%s" % seg["channel"]]["deskew"])
        norms[sid] = core.normalize_segment(seg, ch_deskew)

    # 排除判定：饱和 / 缺样 / 触发在边界
    excluded = {}
    for sid in seg_ids:
        nm = norms[sid]
        reasons = []
        n = len(nm["t"])
        if nm["saturated_idx"] and len(nm["saturated_idx"]) / n > 0.05:
            reasons.append("vertical-saturation")
        if nm["missing_idx"] and len(nm["missing_idx"]) / n > 0.03:
            reasons.append("missing-samples")
        span = nm["t_end"] - nm["t0"]
        edge = max(nm["dt_nominal"], 1e-15)
        if nm["trigger_s"] - nm["t0"] < edge or nm["t_end"] - nm["trigger_s"] < edge \
                or not (nm["t0"] <= nm["trigger_s"] <= nm["t_end"]):
            reasons.append("trigger-at-boundary")
        if reasons:
            excluded[sid] = reasons

    # 多峰对齐候选（参考段：第一段）
    ref_id = seg_ids[0]
    candidates = {}
    for sid in seg_ids:
        cands = core.alignment_candidates(norms[ref_id], norms[sid])
        out = []
        for c in cands:
            cid = "cand-%s-%d" % (sid, c["rank"])
            status = "pending"
            if chosen.get(sid) == cid:
                status = "accepted"
            elif cid in rejected.get(sid, []):
                status = "rejected"
            out.append({"candidate_id": cid, "rank": c["rank"],
                        "lag_s": c["lag_s"], "lag_index": c["lag_index"],
                        "score": c["score"], "status": status})
        candidates[sid] = out

    def chosen_lag(sid):
        want = chosen.get(sid)
        for c in candidates[sid]:
            if want is None and c["rank"] == 1:
                return c["lag_s"]
            if c["candidate_id"] == want:
                return c["lag_s"]
        return 0.0

    # 叠影（对齐后坐标，单位 ns / V，均带坐标基准）
    overlays = {}
    for sid in seg_ids:
        nm = norms[sid]
        lag = chosen_lag(sid)
        aligned = (nm["t"] - nm["t0"] - lag - nm["deskew_s"]) * 1e9
        overlays[sid] = {
            "id": "overlay-%s" % sid,
            "t": {"values": [round(float(x), 6) for x in aligned],
                  "unit": "ns", "basis": "aligned-trigger"},
            "v": {"values": [_nan_to_none(x) for x in nm["v"]],
                  "unit": nm["v_unit"]},
            "valid": [bool(x) for x in nm["valid"]],
            "excluded": sid in excluded,
        }

    # 触发抖动（仅未排除段；deskew 已含通道标定）
    included = [sid for sid in seg_ids if sid not in excluded]
    xs = {sid: norms[sid]["trigger_s"] - norms[sid]["deskew_s"] for sid in included}
    mean_x = float(np.mean(list(xs.values()))) if xs else 0.0
    jitter = []
    for sid in included:
        rules = []
        if not norms[sid]["uniform"]:
            rules.append("nonuniform-timebase-no-index-subtraction")
        jitter.append({
            "id": "jitter-%s" % sid,
            "segment_id": sid,
            "value": xs[sid] - mean_x,
            "unit": "s",
            "basis": "acq_clock",
            "deskew_s": norms[sid]["deskew_s"],
            "rules": rules,
        })

    # 测量窗（区间版本化；缺样/饱和区间不进入插值）
    measurements = []
    for sid in seg_ids:
        nm = norms[sid]
        lag = chosen_lag(sid)
        aligned_t = nm["t"] - nm["t0"] - lag - nm["deskew_s"]
        if sid in excluded:
            meas = {"status": "excluded", "reasons": excluded[sid]}
        else:
            meas = core.measure_window(aligned_t, nm["v"], nm["valid"],
                                       win_start, win_end)
        entry = {
            "id": "meas-%s" % sid,
            "segment_id": sid,
            "interval_version": interval_version,
            "window": {"start": {"value": win_start, "unit": "s", "basis": "aligned"},
                       "end": {"value": win_end, "unit": "s", "basis": "aligned"}},
            "status": meas["status"],
            "rules": ["gap-saturation-excluded-from-interpolation"],
        }
        if meas["status"] == "ok":
            entry["mean"] = {"value": meas["mean"], "unit": nm["v_unit"]}
            entry["min"] = {"value": meas["min"], "unit": nm["v_unit"]}
            entry["max"] = {"value": meas["max"], "unit": nm["v_unit"]}
        else:
            entry["reasons"] = meas.get("reasons", [])
        measurements.append(entry)

    # 派生层依赖（供失效范围计算）
    derived_deps = {}
    for sid in seg_ids:
        seg = src(sid)
        cal = "cal-%s" % seg["channel"]
        derived_deps["align-%s" % sid] = {
            "sources": [sid, ref_id], "params": [],
            "rules": ["multi-peak-alignment-candidates-retained"]}
        derived_deps["overlay-%s" % sid] = {
            "sources": [sid, cal], "params": [], "rules": []}
        derived_deps["jitter-%s" % sid] = {
            "sources": [sid, cal], "params": ["cal-deskew-%s" % seg["channel"]],
            "rules": ([] if norms[sid]["uniform"]
                      else ["nonuniform-timebase-no-index-subtraction"])}
        derived_deps["meas-%s" % sid] = {
            "sources": [sid], "params": [interval_version],
            "rules": ["gap-saturation-excluded-from-interpolation"]}

    segments_out = []
    for sid in seg_ids:
        nm = norms[sid]
        segments_out.append({
            "segment_id": sid,
            "channel": nm["channel"],
            "source_version": active_versions[sid],
            "content_hash": sources[sid]["content_hashes"][active_versions[sid]],
            "n_samples": len(nm["t"]),
            "uniform_timebase": bool(nm["uniform"]),
            "missing_idx": nm["missing_idx"],
            "saturated_count": len(nm["saturated_idx"]),
            "diagnostics": nm["diagnostics"],
            "excluded": sid in excluded,
            "exclude_reasons": excluded.get(sid, []),
        })

    return {
        "branch_id": branch_id,
        "head_seq": head_seq,
        "branches": store.list_branches(conn),
        "active_interval": interval_version,
        "window": {"start": {"value": win_start, "unit": "s", "basis": "aligned"},
                   "end": {"value": win_end, "unit": "s", "basis": "aligned"}},
        "segments": segments_out,
        "candidates": candidates,
        "chosen": {sid: (chosen.get(sid) or "cand-%s-1" % sid) for sid in seg_ids},
        "rejected": rejected,
        "overlays": overlays,
        "jitter": jitter,
        "measurements": measurements,
        "excluded": [{"segment_id": sid, "reasons": r}
                     for sid, r in sorted(excluded.items())],
        "invalidations": invalidations,
        "events": events,
        "derived_deps": derived_deps,
    }


def derived_payloads(state):
    """从状态提取派生层快照（显式保存多候选与裁定相关结论）。"""
    out = {}
    for sid, cands in state["candidates"].items():
        out["align-%s" % sid] = {"candidates": cands}
    for sid, ov in state["overlays"].items():
        out["overlay-%s" % sid] = {"excluded": ov["excluded"]}
    for j in state["jitter"]:
        out[j["id"]] = {"value": j["value"], "unit": j["unit"],
                        "basis": j["basis"]}
    for m in state["measurements"]:
        out[m["id"]] = m
    return out


def refresh_derived(conn, branch_id):
    state = compute_state(conn, branch_id)
    payloads = derived_payloads(state)
    derived = {}
    for key, payload in payloads.items():
        deps = state["derived_deps"].get(key, {"sources": [], "params": [],
                                               "rules": []})
        derived[key] = {"deps": deps, "payload": payload}
    store.replace_derived(conn, branch_id, state["head_seq"], derived)
    return state


def invalidation_scope(state, changed_sources=(), changed_params=()):
    """按依赖关系计算失效范围，不做全局悄悄替换。"""
    affected = []
    for key, deps in state["derived_deps"].items():
        if set(deps["sources"]) & set(changed_sources) or \
                set(deps["params"]) & set(changed_params):
            affected.append(key)
    return sorted(affected)
