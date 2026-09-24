"""Deterministic replay of the event log into domain state.

Projection is a pure function of the event list, so returning to any old
event restores exactly that event's dependency versions and visible
conclusions -- newer computations never overwrite history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from appcore import units as U
from appcore import waveforms as W


def _parse_time(raw, default="ns"):
    try:
        return U.to_seconds(raw, default).in_base(), None
    except U.Diagnostic as d:
        return None, d.as_event()


def _parse_gain(raw):
    try:
        return U.gain_volts_per_count(raw).in_base(), None
    except U.Diagnostic as d:
        return None, d.as_event()


def _parse_probe(raw):
    try:
        return U.attenuation_factor(raw).in_base(), None
    except U.Diagnostic as d:
        return None, d.as_event()


@dataclass
class SegView:
    seg_id: str
    order: int
    label: str
    channels: dict = field(default_factory=dict)
    trigger: dict = field(default_factory=dict)
    included: bool = True
    reason_code: str | None = None
    diagnostics: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    ambiguous: bool = False
    measurements: dict = field(default_factory=dict)
    jitter: dict | None = None
    adopted_candidate: str | None = None
    source_materials: dict = field(default_factory=dict)


def _materialized(channel: dict):
    n = len(channel["samples"])
    axis = W.materialize_time(channel["timebase"], n)
    mask = W.valid_mask(channel["samples"])
    regions = W.mask_regions(channel["samples"], mask)
    gain, gd = _parse_gain(channel.get("vertical_gain"))
    probe, pd = _parse_probe(channel.get("probe_attenuation"))
    deskew, dd = _parse_time(channel.get("deskew", "0 ns"))
    volts = W.to_voltage(channel["samples"], gain, probe, mask)
    diagnostics = list(axis.diagnostics)
    for d in (gd, pd, dd):
        if d:
            diagnostics.append(d)
    for r in regions:
        diagnostics.append({
            "kind": "judgement",
            "code": "saturation-skipped" if r["code"].startswith("saturation")
                    else "gap-skipped",
            "rule": W.RULE_VERSION,
            "message": ("saturated region excluded from interpolation"
                        if r["code"].startswith("saturation")
                        else "missing samples excluded; no interpolation"),
            "region": r,
        })
    return {"time": axis.times, "mask": mask, "regions": regions,
            "gain": gain, "probe": probe, "deskew_s": deskew or 0.0,
            "volts": volts, "diagnostics": diagnostics,
            "voltage_available": gain is not None and probe is not None}


def project(events: list[dict], case: dict) -> dict:
    """Replay ``events`` (ordered) over the immutable case to current state."""
    source = case
    window_version = "v1"
    active_window = case["windows"][0]
    # per-segment channel-level source overrides from patches
    patches: dict[str, dict] = {}
    # rulings: seg_id -> {candidate_id, adopted, event_id, rejected:[...]}
    rulings: dict[str, dict] = {}
    log: list[dict] = []

    for e in events:
        etype = e["type"]
        payload = e.get("payload", {})
        if isinstance(payload, str):
            import json as _json
            payload = _json.loads(payload)
        if etype == "window-revised":
            window_version = payload["to_version"]
            active_window = next(w for w in case["windows"]
                                 if w["version"] == window_version)
            log.append({"event_id": e["event_id"], "type": etype,
                        "detail": payload})
        elif etype == "source-patched":
            key = f"{payload['segment']}/{payload['channel']}"
            patches[key] = payload["change"]
            log.append({"event_id": e["event_id"], "type": etype,
                        "detail": payload,
                        "invalidation": [key]})
        elif etype in ("candidate-ruling",):
            seg = payload["segment"]
            prior = rulings.get(seg, {})
            rejected = list(prior.get("rejected", []))
            if not payload.get("adopted", True) or \
                    payload.get("decision") == "reject":
                if payload["candidate_id"] not in rejected:
                    rejected.append(payload["candidate_id"])
                adopted = prior.get("adopted_candidate")
            else:
                adopted = payload["candidate_id"]
            rulings[seg] = {
                "adopted_candidate": adopted,
                "rejected": rejected,
                "event_id": e["event_id"],
                "note": payload.get("note"),
            }
            log.append({"event_id": e["event_id"], "type": etype,
                        "detail": payload})
        elif etype == "ingest":
            log.append({"event_id": e["event_id"], "type": "ingest",
                        "detail": {"materials": payload.get("materials", {})}})
        else:
            log.append({"event_id": e["event_id"], "type": etype,
                        "detail": payload})

    seg_views: dict[str, SegView] = {}
    pulse = case["reference_pulse"]
    pulse_t = np.asarray(pulse["t"], dtype=float) * 1e-9

    for seg in source["segments"]:
        view = SegView(seg["seg_id"], seg["order"], seg["label"])
        view.trigger = dict(seg["channels"]["trigger"])
        chans = {}
        for cid in ("CH1", "CH2"):
            ch = dict(seg["channels"][cid])
            patch = patches.get(f"{seg['seg_id']}/{cid}")
            patched = False
            if patch:
                for k, v in patch.items():
                    ch[k] = v
                patched = True
            mat = _materialized(ch)
            chans[cid] = mat
            view.source_materials[cid] = {
                "slot": f"{seg['seg_id']}/{cid}",
                "patched": patched,
                "timebase_repr": ch["timebase"],
                "vertical_gain_repr": ch.get("vertical_gain"),
                "probe_attenuation_repr": ch.get("probe_attenuation"),
                "deskew_repr": ch.get("deskew"),
            }
            if cid == "CH2" and abs(mat["deskew_s"]) > 0:
                mat["diagnostics"].append({
                    "kind": "judgement", "code": "channel-deskew-applied",
                    "rule": W.RULE_VERSION,
                    "message": "channel deskew shifts CH2 before alignment",
                    "deskew_s": mat["deskew_s"],
                })
        view.channels = {
            cid: {"n": len(m["time"]),
                  "time_units": "s",
                  "t_start_s": float(np.nanmin(m["time"])),
                  "t_end_s": float(np.nanmax(m["time"])),
                  "voltage_available": m["voltage_available"],
                  "regions": m["regions"]}
            for cid, m in chans.items()
        }
        for cid, m in chans.items():
            for d in m["diagnostics"]:
                d = dict(d)
                d.setdefault("channel", cid)
                view.diagnostics.append(d)

        ref_id = seg["reference_channel"]
        ref = chans[ref_id]

        # --- measurements (active window), valid-only ---
        seg_included = True
        reasons = []
        for cid, m in chans.items():
            win = {"start_s": _parse_time(active_window["start"])[0],
                   "end_s": _parse_time(active_window["end"])[0],
                   "version": window_version}
            meas = W.measure_window(m["time"], m["volts"], m["mask"], win,
                                    regions=m["regions"])
            meas["window"] = {"version": window_version,
                              "start_repr": active_window["start"],
                              "end_repr": active_window["end"]}
            if not m["voltage_available"]:
                meas = {"included": False,
                        "reason_code": "voltage-unconvertible",
                        "valid_fraction": meas.get("valid_fraction", 0.0),
                        "window": meas.get("window", {})}
                meas["window"] = {"version": window_version,
                                  "start_repr": active_window["start"],
                                  "end_repr": active_window["end"]}
            view.measurements[cid] = meas
            if not meas.get("included", False):
                reasons.append(f"{cid}:{meas.get('reason_code')}")

        # --- alignment (multi-peak retained) ---
        # Template normalized to volts at the probe tip (counts -> V via the
        # reference channel's gain and probe attenuation).
        ref_gain = chans[ref_id]["gain"] or 8e-5
        ref_probe = chans[ref_id]["probe"] or 10.0
        pulse_counts = np.asarray(pulse["samples"], dtype=float)
        pulse_sig = pulse_counts * ref_gain * ref_probe
        align_input = {
            cid: {"time": m["time"], "voltage": m["volts"],
                  "mask": m["mask"], "deskew_s": m["deskew_s"]}
            for cid, m in chans.items()
        }
        seg_pulse = {"samples": pulse_sig, "t": pulse_t}
        al = W.align_segment(
            {"seg_id": seg["seg_id"], "reference_channel": ref_id,
             "pulse": seg_pulse},
            align_input)
        view.diagnostics.extend(al.diagnostics)
        view.candidates = [
            {"candidate_id": c.cid, "lag_s": c.lag_s,
             "lag_ns": c.lag_s * 1e9, "score": c.score, "rank": c.rank,
             "adopted": False}
            for c in al.candidates
        ]
        view.ambiguous = al.ambiguous

        ruling = rulings.get(seg["seg_id"])
        if ruling and ruling.get("adopted_candidate"):
            for c in view.candidates:
                c["adopted"] = c["candidate_id"] == \
                    ruling["adopted_candidate"]
            view.adopted_candidate = ruling["adopted_candidate"]
            view.diagnostics.append({
                "kind": "decision", "code": "candidate-adopted",
                "rule": W.RULE_VERSION,
                "candidate_id": ruling["adopted_candidate"],
                "event_id": ruling["event_id"],
                "rejected_candidates": ruling.get("rejected", []),
            })
        if ruling and ruling.get("rejected"):
            view.diagnostics.append({
                "kind": "decision", "code": "candidate-rejected",
                "rule": W.RULE_VERSION,
                "candidate_ids": ruling["rejected"],
                "event_id": ruling["event_id"],
            })

        # --- trigger / jitter ---
        trig_time_s, trig_diag = _parse_time(
            seg["channels"]["trigger"]["timestamp"])
        if trig_diag:
            view.diagnostics.append(trig_diag)
        chosen = next((c for c in view.candidates if c["adopted"]), None)
        lag = chosen["lag_s"] if chosen else (
            view.candidates[0]["lag_s"] if view.candidates else 0.0)
        idx = seg["channels"]["trigger"]["index"]
        if ref["time"] is not None and 0 <= idx < len(ref["time"]):
            trig_axis_s = float(ref["time"][idx])
        else:
            trig_axis_s = None
        event_s = (trig_axis_s + lag) if trig_axis_s is not None else None
        boundary = (idx == 0 or idx >= len(ref["time"]) - 1)
        if boundary:
            view.diagnostics.append({
                "kind": "judgement", "code": "trigger-at-boundary",
                "rule": W.RULE_VERSION,
                "message": "trigger mark at capture edge; jitter excluded, "
                           "waveform kept for review",
                "trigger_index": idx,
            })
        no_peak = any(d.get("code") == "no-alignment-peak"
                      for d in al.diagnostics)
        seg_excluded = bool(reasons)
        jitter_excluded = (boundary or view.ambiguous or no_peak
                           or not view.candidates or seg_excluded)
        jitter = {
            "trigger_timestamp_s": trig_time_s,
            "trigger_axis_s": trig_axis_s,
            "alignment_lag_s": lag,
            "event_s": event_s,
            "included": not jitter_excluded,
            "exclusion_code": ("trigger-at-boundary" if boundary
                               else "ambiguous-multipeak"
                               if view.ambiguous
                               else "alignment-impossible-gap"
                               if no_peak or not view.candidates
                               else "segment-excluded"
                               if seg_excluded else None),
        }
        view.jitter = jitter
        if reasons:
            view.included = False
            view.reason_code = ";".join(reasons)
        seg_views[seg["seg_id"]] = view

    # jitter distribution across included segments, relative to their mean
    evs = [v.jitter["event_s"] for v in seg_views.values()
           if v.included and v.jitter["included"]
           and v.jitter["event_s"] is not None]
    mean_event = float(np.mean(evs)) if evs else None
    for v in seg_views.values():
        if v.jitter and v.jitter["included"] and mean_event is not None:
            v.jitter["jitter_s"] = v.jitter["event_s"] - mean_event
            v.jitter["jitter_ns"] = v.jitter["jitter_s"] * 1e9
        else:
            v.jitter["jitter_s"] = None
            v.jitter["jitter_ns"] = None

    return {
        "rule_version": W.RULE_VERSION,
        "window": {"version": window_version,
                   "start_repr": active_window["start"],
                   "end_repr": active_window["end"]},
        "segments": [_serialize(v) for v in
                     sorted(seg_views.values(), key=lambda x: x.order)],
        "event_log": log,
        "rulings": rulings,
        "mean_event_s": mean_event,
    }


def _serialize(v: SegView) -> dict:
    return {
        "seg_id": v.seg_id, "order": v.order, "label": v.label,
        "included": v.included, "reason_code": v.reason_code,
        "trigger": v.trigger,
        "channels": v.channels,
        "candidates": v.candidates,
        "ambiguous": v.ambiguous,
        "adopted_candidate": v.adopted_candidate,
        "measurements": v.measurements,
        "jitter": v.jitter,
        "diagnostics": v.diagnostics,
        "source_materials": v.source_materials,
    }


def invalidation_for(event_type: str, payload: dict) -> list[str]:
    """Dependency-derived invalidation scope (never global replacement)."""
    if event_type == "source-patched":
        keys = [f"{payload['segment']}/{payload['channel']}"]
        # deskew on CH2 invalidates its own alignment contribution only
        if payload.get("channel") == "CH2":
            keys.append(f"{payload['segment']}/CH2#alignment")
        return keys
    if event_type == "window-revised":
        return ["*/all#measurement-window"]
    if event_type == "candidate-ruling":
        return [f"{payload['segment']}#ruling",
                f"{payload['segment']}#jitter"]
    if event_type == "deskew-revised":
        return [f"{payload['segment']}/CH2#alignment",
                f"{payload['segment']}#jitter"]
    return ["*"]
