"""Ingest segmented captures into immutable source materials.

On failure the whole change rolls back -- no half records remain. Raw bytes
are stored verbatim; only the derivation layer normalizes units.
"""
from __future__ import annotations

import json

from appcore.casegen import build_case
from appcore.repository import Repository, stable_id

SLOT_FIELDS = ("samples", "timebase", "vertical_gain",
               "probe_attenuation", "trigger", "deskew")


def validate_case(case: dict) -> list[str]:
    errors = []
    for key in ("case_id", "segments", "windows", "reference_pulse"):
        if key not in case:
            errors.append(f"missing top-level field: {key}")
    for seg in case.get("segments", []):
        for key in ("seg_id", "channels"):
            if key not in seg:
                errors.append(f"segment missing {key}")
        for cid in ("CH1", "CH2"):
            ch = seg.get("channels", {}).get(cid)
            if ch is None:
                errors.append(f"{seg.get('seg_id')}: missing channel {cid}")
                continue
            for f in ("samples", "timebase", "vertical_gain",
                      "probe_attenuation", "deskew"):
                if f not in ch:
                    errors.append(f"{seg['seg_id']}/{cid}: missing {f}")
        if "trigger" not in seg.get("channels", {}):
            errors.append(f"{seg.get('seg_id')}: missing trigger mark")
    return errors


def ingest_case(repo: Repository, case: dict,
                branch_id: str = "main") -> dict:
    """Atomic ingest: materials + one ingest event, all-or-nothing."""
    errors = validate_case(case)
    if errors:
        raise ValueError("case validation failed: " + "; ".join(errors))
    repo.ensure_main_branch()

    materials: dict[str, str] = {}
    try:
        with repo.transaction():
            root = repo.put_material(f"case/{case['case_id']}/root", {
                "case_id": case["case_id"], "title": case["title"],
                "coordinate_basis": case.get("coordinate_basis", {}),
            })
            materials["root"] = root
            pulse = repo.put_material(
                f"case/{case['case_id']}/reference_pulse",
                case["reference_pulse"])
            materials["reference_pulse"] = pulse
            windows = repo.put_material(
                f"case/{case['case_id']}/windows",
                case["windows"])
            materials["windows"] = windows
            for seg in case["segments"]:
                sid = seg["seg_id"]
                for cid in ("CH1", "CH2"):
                    ch = seg["channels"][cid]
                    for f in SLOT_FIELDS:
                        if f == "trigger":
                            continue
                        mid = repo.put_material(
                            f"source/{sid}/{cid}/{f}",
                            {"seg_id": sid, "channel": cid, "field": f,
                             "original": ch[f]})
                        materials[f"{sid}/{cid}/{f}"] = mid
                trig = seg["channels"]["trigger"]
                mid = repo.put_material(f"source/{sid}/trigger", {
                    "seg_id": sid, "field": "trigger", "original": trig})
                materials[f"{sid}/trigger"] = mid
            payload = {
                "case_id": case["case_id"],
                "title": case["title"],
                "rule_version": __import__("appcore.waveforms",
                                           fromlist=["x"]).RULE_VERSION,
                "materials": materials,
                "segment_count": len(case["segments"]),
            }
            event = repo.append(branch_id, "ingest", payload)
    except Exception:
        raise
    return {"event": event, "materials": materials}


def bundled_case() -> dict:
    return build_case()
