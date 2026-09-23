"""派生层：对齐 / 抖动 / 测量窗。

硬性规则（违反即 bug，需产出机器可读判定事件）：
  R1 时间轴不等间隔时不用索引相减 —— 一律按真实时间坐标计算。
  R2 缺样与饱和区间不进入插值 —— 插值与相关都只使用有效样本。
  R3 对齐保留多峰互相关候选 —— 不擅自收敛到单峰。
所有领域量携带单位或坐标基准；本层把原始表示规范化为 SI（秒/伏）。
"""
import numpy as np

from .units import to_si

RULE_VERSION = "align-rules:v1"
EDGE_MARGIN_SAMPLES = 2
MIN_OVERLAP = 32


# ---------------------------------------------------------------- 段视图
def build_segment_view(raw):
    """把原始段 JSON 规范化为 SI 视图。无法转换时保留 None 并给出诊断事件。"""
    seg_id = raw["id"]
    rule_events = []
    tb = raw["timebase"]
    n = len(raw["samples"])
    kind = tb.get("kind")
    t = None
    if kind == "uniform":
        t0, e1 = to_si(tb.get("t0"), "time")
        dt, e2 = to_si(tb.get("dt"), "time")
        if e1 or e2:
            rule_events.append({"rule": "diagnostic", "segment": seg_id,
                                "detail": f"timebase-unconvertible:{e1 or ''}{e2 or ''}"})
        else:
            t = t0 + dt * np.arange(n)
    elif kind == "nonuniform":
        unit = tb.get("t_unit", "")
        probe, err = to_si({"value": 1.0, "unit": unit}, "time")
        if err:
            rule_events.append({"rule": "diagnostic", "segment": seg_id,
                                "detail": f"timebase-unconvertible:{err}"})
        else:
            t = np.asarray(tb["t"], dtype=float) * probe
            # R1: 不等间隔时间轴 —— 记录判定事件，禁止后续用索引差代替时间差
            rule_events.append({
                "rule": "nonuniform-no-index-sub", "segment": seg_id,
                "detail": "时间轴不等间隔，全部时延按真实时间坐标计算，不用索引相减"})
    if t is None:
        t = np.zeros(n)  # 占位，视图标记为不可用
    dt_median = float(np.median(np.diff(t))) if n > 1 else 0.0

    v = None
    factor, err = to_si({"value": 1.0, "unit": raw.get("sample_unit", "")}, "voltage")
    if err:
        rule_events.append({"rule": "diagnostic", "segment": seg_id,
                            "detail": f"samples-unconvertible:{err}"})
    else:
        atten, aerr = to_si(raw.get("probe_attenuation"), "ratio")
        if aerr:
            rule_events.append({"rule": "diagnostic", "segment": seg_id,
                                "detail": f"attenuation-unconvertible:{aerr}"})
            atten = 1.0
        v = np.asarray(raw["samples"], dtype=float) * factor * atten

    valid = np.ones(n, dtype=bool)
    for start, count in raw.get("missing", []):
        valid[int(start):int(start) + int(count)] = False
    if v is not None:
        fs, ferr = to_si(raw.get("full_scale"), "voltage")
        if not ferr and fs:
            atten, _ = to_si(raw.get("probe_attenuation"), "ratio")
            valid &= np.abs(v) < abs(fs * (atten or 1.0))
    n_invalid = int((~valid).sum())
    if n_invalid:
        # R2: 缺样与饱和区间不进入插值
        rule_events.append({
            "rule": "gap-no-interp", "segment": seg_id,
            "detail": f"{n_invalid} 个缺样/饱和样本被排除在插值与相关之外"})

    trig, terr = to_si(raw.get("trigger", {}).get("timestamp"), "time")
    if terr:
        trig = None
        rule_events.append({"rule": "diagnostic", "segment": seg_id,
                            "detail": f"trigger-unconvertible:{terr}"})
    deskew, derr = to_si(raw.get("deskew"), "time")
    if derr:
        deskew = 0.0
        rule_events.append({"rule": "diagnostic", "segment": seg_id,
                            "detail": f"deskew-unconvertible:{derr}"})

    return {
        "id": seg_id, "channel": raw["channel"], "kind": kind, "n": n,
        "t": t, "dt": dt_median, "v": v, "valid": valid,
        "trigger": trig, "deskew": deskew,
        "raw": raw, "rule_events": rule_events,
    }


def invalid_intervals(view):
    """有效样本之间的空洞（缺样/饱和）区间，插值查询点落入即排除。"""
    t, valid = view["t"], view["valid"]
    intervals = []
    idx = np.where(~valid)[0]
    if len(idx) == 0:
        return intervals
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for g in groups:
        lo = t[max(g[0] - 1, 0)]
        hi = t[min(g[-1] + 1, len(t) - 1)]
        intervals.append((float(lo), float(hi)))
    return intervals


# ---------------------------------------------------------------- 对齐
def _masked_interp(xp, fp, gaps, x):
    """仅在有效样本上插值；落入空洞区间的查询点被掩掉（R2）。"""
    y = np.interp(x, xp, fp, left=np.nan, right=np.nan)
    mask = ~np.isnan(y)
    for lo, hi in gaps:
        mask &= ~((x >= lo) & (x <= hi))
    return y, mask


def correlation_candidates(ref, seg, params):
    """计算 seg 相对 ref 的时延候选（秒）。保留所有超过阈值比例的峰（R3）。"""
    max_lag = params["max_lag_s"]
    threshold = params["peak_threshold"]
    step = min(ref["dt"] or max_lag / 50, seg["dt"] or max_lag / 50) / 2.0
    if step <= 0:
        step = max_lag / 100.0
    lags = np.arange(-max_lag, max_lag + step, step)

    sv, tv = seg["v"], seg["t"]
    smask = seg["valid"] & np.isfinite(sv)
    ref_gaps = invalid_intervals(ref)
    rv, rt = ref["v"], ref["t"]
    rmask = ref["valid"] & np.isfinite(rv)
    rp, fp = rt[rmask], rv[rmask]

    scores = np.full(len(lags), -np.inf)
    st, svv = tv[smask], sv[smask]
    svv = svv - svv.mean()
    snorm = np.sqrt(np.sum(svv ** 2)) or 1.0
    for i, lag in enumerate(lags):
        y, m = _masked_interp(rp, fp, ref_gaps, st + lag)
        if m.sum() < MIN_OVERLAP:
            continue
        a = svv[m]
        b = y[m] - y[m].mean()
        bnorm = np.sqrt(np.sum(b ** 2))
        if bnorm == 0:
            continue
        scores[i] = float(np.dot(a, b) / (snorm * bnorm))

    best = scores.max()
    if not np.isfinite(best):
        return [], scores, lags
    cands = []
    for i in range(1, len(lags) - 1):
        if (scores[i] >= scores[i - 1] and scores[i] > scores[i + 1]
                and scores[i] >= threshold * best):
            cands.append({"tau": float(lags[i]), "score": float(scores[i])})
    cands.sort(key=lambda c: -c["score"])
    for rank, c in enumerate(cands):
        c["rank"] = rank
    return cands, scores, lags


def align_segments(views, params):
    """对全部段做对齐。返回 {seg_id: {candidates, rule_events}}。参考段 = 第一段。"""
    ref = views[0]
    out = {}
    for view in views:
        entry = {"candidates": [], "rule_events": list(view["rule_events"])}
        if view["v"] is None:
            entry["rule_events"].append({
                "rule": "diagnostic", "segment": view["id"],
                "detail": "样本单位无法转换，段保留在原位置，不参与对齐"})
            out[view["id"]] = entry
            continue
        if view["id"] == ref["id"]:
            entry["candidates"] = [{"tau": 0.0, "score": 1.0, "rank": 0}]
        else:
            cands, _, _ = correlation_candidates(ref, view, params)
            entry["candidates"] = [
                {"id": f"cand:{view['id']}/{c['rank']}", **c} for c in cands]
            if len(cands) > 1:
                entry["rule_events"].append({
                    "rule": "multipeak-candidates", "segment": view["id"],
                    "detail": f"互相关出现 {len(cands)} 个合法峰，全部保留为候选"})
        out[view["id"]] = entry
    return out


# ---------------------------------------------------------------- 排除
def exclusion_reasons(view):
    """触发在边界等结构性排除。返回原因列表（机器可读）。"""
    reasons = []
    if view["trigger"] is None or view["v"] is None:
        return reasons
    t = view["t"]
    pos = float(np.searchsorted(t, view["trigger"]))
    if pos <= EDGE_MARGIN_SAMPLES or pos >= view["n"] - 1 - EDGE_MARGIN_SAMPLES:
        reasons.append({
            "rule": "trigger-at-boundary", "segment": view["id"],
            "detail": f"触发时标位于段边界（位置 {pos:.1f}/{view['n']}），段被排除"})
    return reasons


# ---------------------------------------------------------------- 抖动与测量
def selected_candidate(align_entry, rulings, seg_id):
    """根据裁定选择候选：显式 accept 优先；全部被 reject 则返回 None；否则 rank0。"""
    cands = align_entry.get("candidates", [])
    if not cands:
        return None
    for c in cands:
        r = rulings.get(c.get("id", ""))
        if r == "accept":
            return c
    alive = [c for c in cands if rulings.get(c.get("id", "")) != "reject"]
    if not alive:
        return None
    return alive[0]


def compute_view(views, alignments, rulings, intervals, ref_id):
    """汇总叠影、抖动分布、测量窗、排除段。所有输出携带单位基准（秒/伏）。"""
    ref = views[0]
    ref_event = (ref["trigger"] or 0.0) - ref["deskew"]
    segs, jitter, excluded = [], {}, []
    for view in views:
        reasons = exclusion_reasons(view)
        align = alignments[view["id"]]
        cand = selected_candidate(align, rulings, view["id"])
        manual = rulings.get(f"seg:{view['id']}")
        if manual == "exclude":
            reasons = reasons + [{"rule": "manual-exclude", "segment": view["id"],
                                  "detail": "人工裁定排除"}]
        if manual == "include":
            reasons = [r for r in reasons if r["rule"] != "trigger-at-boundary"]
        usable = view["v"] is not None
        if reasons or not usable:
            excluded.append({"segment": view["id"], "reasons": reasons or [
                {"rule": "diagnostic", "segment": view["id"],
                 "detail": "单位无法转换，段保留在原位置"}]})
        tau = cand["tau"] if cand else 0.0
        event = (view["trigger"] or 0.0) - view["deskew"] + tau
        jitter[view["id"]] = event - ref_event
        item = {"id": view["id"], "channel": view["channel"],
                "excluded": bool(reasons) or not usable,
                "tau": tau, "selected": cand,
                "candidates": align["candidates"],
                "unit": {"t": "s", "v": "V"}}
        if usable:
            item["t"] = (view["t"] + tau - ref_event).tolist()
            item["v"] = view["v"].tolist()
            item["valid"] = view["valid"].tolist()
        segs.append(item)

    inc = [jitter[s["id"]] for s in segs if not s["excluded"]]
    jitter_stats = {"unit": "s", "count": len(inc)}
    if inc:
        arr = np.asarray(inc)
        jitter_stats.update({"mean": float(arr.mean()), "std": float(arr.std()),
                             "min": float(arr.min()), "max": float(arr.max())})

    windows = []
    for w in intervals.get("windows", []):
        t0, _ = to_si(w["t0"], "time")
        t1, _ = to_si(w["t1"], "time")
        windows.append({"id": w["id"], "t0": t0, "t1": t1, "unit": "s"})

    measurements = {}
    for s, view in zip(segs, views):
        if s["excluded"] or view["v"] is None:
            continue
        tau = s["tau"]
        t_al = view["t"] + tau - ref_event
        per = {}
        for w in windows:
            m = (t_al >= w["t0"]) & (t_al <= w["t1"])
            vals = view["v"][m & view["valid"]]
            dropped = int((m & ~view["valid"]).sum())
            mid = f"meas:{view['id']}/{w['id']}@{intervals.get('version', 'v?')}"
            if len(vals) == 0:
                per[w["id"]] = {"id": mid, "status": "empty", "unit": "V",
                                "excluded_samples": dropped}
            else:
                per[w["id"]] = {
                    "id": mid, "status": "partial" if dropped else "ok", "unit": "V",
                    "mean": float(vals.mean()), "min": float(vals.min()),
                    "max": float(vals.max()), "count": int(len(vals)),
                    "excluded_samples": dropped}
        measurements[view["id"]] = per

    rule_events = []
    for a in alignments.values():
        rule_events.extend(a["rule_events"])
    for e in excluded:
        rule_events.extend(e["reasons"])

    return {"segments": segs, "jitter": jitter, "jitter_stats": jitter_stats,
            "windows": windows, "measurements": measurements,
            "excluded": excluded, "rule_events": rule_events,
            "ref_event": ref_event, "unit": {"t": "s", "v": "V", "jitter": "s"}}
