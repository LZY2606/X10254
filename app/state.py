"""Event-sourced review state. Source facts and derived interpretations live
apart: events only ever append; re-interpretation opens a new branch; old
branches, decisions and the raw bytes they reference stay readable."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import derivation as der
from . import storage

MAIN_BRANCH = "br:main"


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------

def reduce_events(events: list[dict]) -> dict:
    fact_versions: dict[str, str] = {}
    decisions: dict[str, dict] = {}
    case_meta: dict = {}
    for event in events:
        payload = json.loads(event["payload_json"])
        etype = event["event_type"]
        if etype == "import_case":
            case_meta = {"case_id": payload["case_id"], "title": payload["title"]}
            for fact in payload["facts"]:
                fact_versions[fact["fact_key"]] = fact["fact_id"]
        elif etype in ("revise_source", "revise_calibration",
                       "revise_window", "revise_params"):
            fact_versions[payload["fact_key"]] = payload["new_fact_id"]
        elif etype == "decide":
            decisions[payload["candidate_id"]] = {
                "status": payload["status"],
                "event_id": event["event_id"],
                "seq": event["seq"],
                "note": payload.get("note"),
            }
        elif etype == "reset_decision":
            decisions.pop(payload["candidate_id"], None)
    return {"fact_versions": fact_versions, "decisions": decisions,
            "case_meta": case_meta}


def state_hash(reduced: dict) -> str:
    blob = json.dumps({
        "fact_versions": dict(sorted(reduced["fact_versions"].items())),
        "decisions": {k: v["status"] for k, v in
                      sorted(reduced["decisions"].items())},
        "rules": der.RULE_VERSION,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Derived state
# ---------------------------------------------------------------------------

def fact_node(fact_key: str, version: int) -> str:
    return f"fact:{fact_key}@v{version}"


def derive(conn, branch_id: str, reduced: dict) -> dict:
    facts_by_id = {}
    for fact in storage.facts(conn):
        facts_by_id[fact["fact_id"]] = fact

    def current(fact_key):
        fact_id = reduced["fact_versions"].get(fact_key)
        return facts_by_id.get(fact_id) if fact_id else None

    # gather current facts by type
    segments, calibs, windows = {}, {}, {}
    params = None
    for key, fact_id in sorted(reduced["fact_versions"].items()):
        fact = facts_by_id[fact_id]
        payload = json.loads(fact["payload_json"])
        if fact["fact_type"] == "segment":
            segments[payload["segment_id"]] = (fact, payload)
        elif fact["fact_type"] == "calibration":
            calibs[payload["channel_id"]] = (fact, payload)
        elif fact["fact_type"] == "window":
            windows[payload["window_id"]] = (fact, payload)
        elif fact["fact_type"] == "params":
            params = (fact, payload)

    grid = der.ref_grid()
    edges: list[tuple] = []
    findings: list[dict] = []
    views: dict[str, dict] = {}

    def add_finding(code, subject, adopted, detail):
        findings.append({
            "finding_id": f"fnd:{code}:{subject}",
            "code": code, "subject": subject, "adopted": bool(adopted),
            "detail": detail, "rule_version": der.RULE_VERSION,
        })

    for seg_id, (fact, payload) in sorted(segments.items()):
        channel = payload.get("channel_id", "CH1")
        calib_fact, calib_payload = calibs.get(channel, (None, {}))
        view = der.interpret_segment(payload, calib_payload)
        view["fact_id"] = fact["fact_id"]
        view["fact_version"] = fact["version"]
        view["wave_id"] = f"wave:{seg_id}@{fact['fact_id']}"
        views[seg_id] = view
        edges.append((fact_node(fact["fact_key"], fact["version"]),
                      view["wave_id"], "interprets"))
        if calib_fact is not None:
            edges.append((fact_node(calib_fact["fact_key"],
                                    calib_fact["version"]),
                          view["wave_id"], "calibrates"))
        if view["irregular"]:
            add_finding("timebase_nonuniform", view["wave_id"], True,
                        "physical timestamps used; index subtraction forbidden")
        if view["saturated"]:
            add_finding("saturated_excluded", view["wave_id"], True,
                        "saturated runs excluded from interpolation")
        if view["missing"]:
            add_finding("missing_excluded", view["wave_id"], True,
                        "missing samples excluded from interpolation")
        if view["boundary"]:
            add_finding("trigger_boundary", view["wave_id"], True,
                        "trigger mark on record boundary")
        if not view["usable"]:
            add_finding("unconvertible_quantity", view["wave_id"], True,
                        "raw value kept in place; not normalized")

    ref_view = views.get(der.REF_TRIGGER_ID)
    ref_wave = der.resample(ref_view, grid) if ref_view else None
    ref_trigger = ref_view["trigger_s"] if ref_view else None

    candidates: dict[str, list[dict]] = {}
    chosen: dict[str, dict] = {}
    jitter_entries: list[dict] = []
    overlay: list[dict] = []

    for seg_id, view in sorted(views.items()):
        if not view["usable"]:
            continue
        wave = der.resample(view, grid)
        valid = ~np_isnan(wave)
        overlay.append({
            "segment_id": seg_id, "wave_id": view["wave_id"],
            "channel_id": view["channel_id"],
            "t_ns": [round(float(t) * 1e9, 6) for t in grid],
            "volts": [None if v != v else round(float(v), 6) for v in wave],
        })
        if ref_wave is None:
            continue
        cands = [] if seg_id == der.REF_TRIGGER_ID else der.candidate_lags(
            wave, ref_wave, valid)
        if seg_id == der.REF_TRIGGER_ID:
            cands = [{"lag_ns": 0.0, "score": 1.0}]
        cand_objs = []
        for cand in cands:
            cand_id = (f"cand:{seg_id}@{view['fact_id']}"
                       f"#{cand['lag_ns']:+.1f}ns")
            decision = reduced["decisions"].get(cand_id)
            cand_objs.append({
                "candidate_id": cand_id, "segment_id": seg_id,
                "lag_ns": cand["lag_ns"], "score": cand["score"],
                "status": decision["status"] if decision else "pending",
                "decision_event_id": decision["event_id"] if decision else None,
            })
            edges.append((view["wave_id"], cand_id, "aligns"))
        candidates[seg_id] = cand_objs
        if len(cand_objs) > 1:
            for cand in cand_objs:
                add_finding("multi_peak_candidate", cand["candidate_id"],
                            cand["status"] == "accepted",
                            {"lag_ns": cand["lag_ns"], "score": cand["score"],
                             "segment_id": seg_id})
        pick = _pick(cand_objs)
        if pick:
            chosen[seg_id] = pick
            calib_fact, _ = calibs.get(view["channel_id"], (None, {}))
            jit_id = (f"jit:{seg_id}@{view['fact_id']}"
                      f"@{calib_fact['fact_id'] if calib_fact else 'nocalib'}")
            jitter = der.jitter_ns(pick["lag_ns"], view, ref_trigger)
            reason = _exclusion_reason(view, cand_objs, pick)
            edges.append((pick["candidate_id"], jit_id, "jitter"))
            if calib_fact is not None:
                edges.append((fact_node(calib_fact["fact_key"],
                                        calib_fact["version"]),
                              jit_id, "deskew"))
            jitter_entries.append({
                "jitter_id": jit_id, "segment_id": seg_id,
                "candidate_id": pick["candidate_id"],
                "lag_ns": pick["lag_ns"],
                "jitter_ns": None if jitter is None else round(jitter, 6),
                "excluded": reason is not None, "reason": reason,
                "trigger_raw": view["trigger_raw"],
                "deskew_s": view["deskew_s"],
            })

    measurements: list[dict] = []
    for window_id, (wfact, wpayload) in sorted(windows.items()):
        q0 = der.parse_quantity(wpayload["t0"], "time")
        q1 = der.parse_quantity(wpayload["t1"], "time")
        for seg_id, view in sorted(views.items()):
            if not view["usable"]:
                continue
            meas_id = (f"meas:{window_id}@{wfact['fact_id']}"
                       f":{seg_id}@{view['fact_id']}")
            edges.append((fact_node(wfact["fact_key"], wfact["version"]),
                          meas_id, "window"))
            edges.append((views[seg_id]["wave_id"], meas_id, "measures"))
            wave = der.resample(view, grid)
            t_ns = grid * 1e9
            if q0.ok and q1.ok:
                mask = (grid >= q0.value) & (grid <= q1.value) & ~np_isnan(wave)
            else:
                mask = np.zeros(len(grid), dtype=bool)
            vals = wave[mask]
            measurements.append({
                "meas_id": meas_id, "window_id": window_id,
                "window_version": wfact["version"],
                "segment_id": seg_id,
                "t0_raw": wpayload["t0"], "t1_raw": wpayload["t1"],
                "n_valid": int(vals.size),
                "mean_v": None if vals.size == 0 else round(float(vals.mean()), 6),
                "max_v": None if vals.size == 0 else round(float(vals.max()), 6),
                "min_v": None if vals.size == 0 else round(float(vals.min()), 6),
                "excluded": bool(vals.size == 0),
                "reason": None if vals.size else "no_valid_samples_in_window",
            })

    if params is not None:
        pfact, _ = params
        for seg_id, cand_objs in candidates.items():
            for cand in cand_objs:
                edges.append((fact_node(pfact["fact_key"], pfact["version"]),
                              cand["candidate_id"], "params"))

    state = {
        "branch_id": branch_id,
        "state_hash": state_hash(reduced),
        "rule_version": der.RULE_VERSION,
        "case": reduced["case_meta"],
        "fact_versions": dict(reduced["fact_versions"]),
        "segments": _segment_summaries(views, candidates, jitter_entries),
        "overlay": overlay,
        "candidates": candidates,
        "jitter": jitter_entries,
        "measurements": measurements,
        "windows": [
            {"window_id": wid, "fact_id": f["fact_id"],
             "version": f["version"],
             "label": json.loads(f["payload_json"]).get("label", wid),
             "t0": json.loads(f["payload_json"])["t0"],
             "t1": json.loads(f["payload_json"])["t1"]}
            for wid, (f, _) in sorted(windows.items())],
        "findings": findings,
        "edges": edges,
        "decisions": reduced["decisions"],
    }
    storage.put_findings(conn, branch_id, state["state_hash"], findings)
    storage.put_edges(conn, branch_id, state["state_hash"], edges)
    return state


def _pick(cand_objs: list[dict]) -> dict | None:
    accepted = [c for c in cand_objs if c["status"] == "accepted"]
    if accepted:
        return accepted[0]
    if len(cand_objs) == 1 and cand_objs[0]["status"] != "rejected":
        return cand_objs[0]
    return None


def _exclusion_reason(view, cand_objs, pick) -> str | None:
    if view["boundary"]:
        return "trigger_on_boundary"
    if view["saturated"]:
        return "saturated_run"
    if view["missing"]:
        return "missing_samples"
    if len(cand_objs) > 1 and pick["status"] != "accepted":
        return "ambiguous_multi_peak"
    return None


def _segment_summaries(views, candidates, jitter_entries) -> list[dict]:
    jitter_by_seg = {j["segment_id"]: j for j in jitter_entries}
    out = []
    for seg_id, view in sorted(views.items()):
        out.append({
            "segment_id": seg_id,
            "wave_id": view["wave_id"],
            "channel_id": view["channel_id"],
            "fact_id": view["fact_id"],
            "fact_version": view["fact_version"],
            "usable": view["usable"],
            "flags": {
                "irregular_timebase": view["irregular"],
                "saturated": view["saturated"],
                "missing": view["missing"],
                "trigger_boundary": view["boundary"],
            },
            "trigger_raw": view["trigger_raw"],
            "n_candidates": len(candidates.get(seg_id, [])),
            "jitter": jitter_by_seg.get(seg_id),
            "diag": view["diag"],
        })
    return out


# ---------------------------------------------------------------------------
# Invalidation scope (dependency graph, never a global silent replace)
# ---------------------------------------------------------------------------

def impact_of(state: dict, fact_key: str) -> list[str]:
    fact_id = state["fact_versions"].get(fact_key)
    if fact_id is None:
        return []
    version = int(fact_id.rsplit(":v", 1)[1])
    start = fact_node(fact_key, version)
    adj: dict[str, list[str]] = {}
    for src, dst, _kind in state["edges"]:
        adj.setdefault(src, []).append(dst)
    seen, queue = set(), [start]
    while queue:
        node = queue.pop()
        for nxt in adj.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return sorted(seen)


def state_at(conn, branch_id: str, upto_seq: int | None = None) -> dict:
    events = storage.events(conn, branch_id, upto_seq)
    reduced = reduce_events(events)
    state = derive(conn, branch_id, reduced)
    state["upto_seq"] = upto_seq if upto_seq is not None else (
        events[-1]["seq"] if events else 0)
    state["events"] = [
        {"seq": e["seq"], "event_id": e["event_id"],
         "event_type": e["event_type"], "payload": json.loads(e["payload_json"])}
        for e in events]
    return state
