"""Import segmented captures into the source layer.

Source and interpretation are stored apart. Every segment fact references the
exact original bytes (a blob hash). A failed import rolls back; a partial
record set can never be left behind.
"""
from __future__ import annotations

import hashlib
import json
import os

from . import storage
from .derivation import short_hash

CASES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "cases", "waveform-overlay-lineage")


def load_case_dir(case_dir: str = CASES_DIR) -> dict:
    with open(os.path.join(case_dir, "case.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    segments = []
    for segment_id in manifest["segments"]:
        with open(os.path.join(case_dir, f"{segment_id}.json"),
                  encoding="utf-8") as f:
            segments.append(json.load(f))
    with open(os.path.join(case_dir, "windows.json"), encoding="utf-8") as f:
        windows = json.load(f)["windows"]
    return {"manifest": manifest, "segments": segments, "windows": windows}


def _fact_id(fact_type: str, fact_key: str, version: int, content) -> str:
    digest = short_hash(json.dumps(content, sort_keys=True, ensure_ascii=False), 12)
    return f"{fact_type}:{fact_key}:v{version}:{digest}"


def import_case(conn, branch_id: str, case: dict, event_id: str, seq: int,
                version: int = 1) -> dict:
    """Write all source facts atomically. Caller owns the transaction."""
    manifest, segments, windows = case["manifest"], case["segments"], case["windows"]
    fact_refs = []

    def add(fact_type, key, payload, blob_bytes=None):
        fid = _fact_id(fact_type, key, version, payload)
        blob_hash = None
        if blob_bytes is not None:
            blob_hash = hashlib.sha256(blob_bytes).hexdigest()
            storage.put_blob(conn, blob_hash, blob_bytes)
        storage.insert_fact(conn, fid, fact_type, key, version, payload,
                            blob_hash, event_id, seq)
        fact_refs.append({"fact_key": key, "fact_id": fid})
        return fid

    params_fid = add("params", "params:alignment", {
        **manifest.get("params", {}), "rule_version": "rules-2026.09"})

    for channel in manifest["channels"]:
        add("calibration", f"calib:{channel['channel_id']}", {
            "channel_id": channel["channel_id"],
            "label": channel.get("label", channel["channel_id"]),
            "vertical_gain": channel["vertical_gain"],
            "probe_attenuation": channel["probe_attenuation"],
            "deskew": channel["deskew"],
            "version_note": "import v1",
        })

    for segment in segments:
        raw_bytes = json.dumps(
            segment, sort_keys=True, ensure_ascii=False).encode("utf-8")
        add("segment", f"seg:{segment['segment_id']}", segment, raw_bytes)

    for window in windows:
        add("window", f"win:{window['window_id']}", {
            **window, "version_note": "v1 initial"})

    storage.append_event(conn, branch_id, event_id, "import_case", {
        "case_id": manifest["case_id"], "title": manifest["title"],
        "facts": fact_refs, "params_fact_id": params_fid,
        "n_segments": len(segments), "n_windows": len(windows),
    })
    return {"fact_refs": fact_refs, "params_fact_id": params_fid}


def revise_fact(conn, branch_id, event_type, fact_type, fact_key, payload,
                seq, event_id, reason, preserve_blob=False, blob_bytes=None):
    """Append a new version of one direct source fact. Never overwrites the
    old version, so prior branches remain readable."""
    existing = conn.execute(
        "SELECT version FROM facts WHERE fact_key=? ORDER BY version DESC LIMIT 1",
        (fact_key,)).fetchone()
    version = (existing["version"] + 1) if existing else 1
    fid = _fact_id(fact_type, fact_key, version, payload)
    blob_hash = None
    if preserve_blob and blob_bytes is not None:
        blob_hash = hashlib.sha256(blob_bytes).hexdigest()
        storage.put_blob(conn, blob_hash, blob_bytes)
    storage.insert_fact(conn, fid, fact_type, fact_key, version, payload,
                        blob_hash, event_id, seq)
    return fid
