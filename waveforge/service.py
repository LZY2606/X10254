from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import domain as D

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "case_pack" / "waveform_overlap_station"
DEFAULT_DB = ROOT / "waveforge_review.sqlite3"

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS branches (
  branch_id TEXT PRIMARY KEY,
  parent_branch_id TEXT REFERENCES branches(branch_id),
  label TEXT NOT NULL,
  parent_event_ord INTEGER,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  ord INTEGER NOT NULL UNIQUE,
  branch_id TEXT NOT NULL REFERENCES branches(branch_id),
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_blobs (
  blob_id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  content BLOB NOT NULL,
  sha256 TEXT NOT NULL,
  imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event_blobs (
  event_id TEXT NOT NULL REFERENCES events(event_id),
  blob_id TEXT NOT NULL REFERENCES raw_blobs(blob_id),
  role TEXT NOT NULL,
  PRIMARY KEY(event_id, blob_id, role)
);
CREATE TABLE IF NOT EXISTS counterexamples (
  counterexample_id TEXT PRIMARY KEY,
  payload_json TEXT NOT NULL,
  adopted_event_id TEXT NOT NULL,
  not_adopted_event_id TEXT NOT NULL,
  machine_judgment_event_id TEXT NOT NULL
);
"""


class ConflictError(Exception):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class ImportErrorRollback(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def insert_blob(conn: sqlite3.Connection, event_id: str, path: Path, role: str) -> str:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    blob_id = f"blob:{path.name}:{digest[:16]}"
    conn.execute(
        "INSERT OR IGNORE INTO raw_blobs(blob_id,filename,content,sha256,imported_at) VALUES(?,?,?,?,?)",
        (blob_id, path.name, content, digest, now()),
    )
    conn.execute(
        "INSERT OR IGNORE INTO event_blobs(event_id,blob_id,role) VALUES(?,?,?)",
        (event_id, blob_id, role),
    )
    return blob_id


def next_ord(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(ord),0) + 1 FROM events").fetchone()
    return int(row[0])


def make_event_id(ord_value: int, kind: str, digest: str) -> str:
    return f"evt:{ord_value:05d}:{kind}:{digest[:12]}"


def append_event(conn: sqlite3.Connection, branch_id: str, kind: str, payload: dict[str, Any],
                 blobs: list[tuple[Path, str]] | None = None) -> dict[str, Any]:
    init_db(conn)
    ord_value = next_ord(conn)
    stored = dict(payload)
    stored.update({"event_ord": ord_value, "branch_id": branch_id})
    digest = canonical_hash(stored)
    event_id = make_event_id(ord_value, kind, digest)
    conn.execute(
        "INSERT INTO events(event_id,ord,branch_id,kind,payload_json,created_at,sha256) VALUES(?,?,?,?,?,?,?)",
        (event_id, ord_value, branch_id, kind, json.dumps(payload, ensure_ascii=False, sort_keys=True),
         now(), digest),
    )
    for path, role in blobs or []:
        insert_blob(conn, event_id, path, role)
    return {"event_id": event_id, "ord": ord_value, "kind": kind, "branch_id": branch_id,
            "payload": payload, "sha256": digest}


def ensure_branch(conn: sqlite3.Connection, branch_id: str = "MAIN", label: str = "main review",
                  parent: str | None = None, parent_event_ord: int | None = None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO branches(branch_id,parent_branch_id,label,parent_event_ord,created_at) VALUES(?,?,?,?,?)",
        (branch_id, parent, label, parent_event_ord, now()),
    )


def initial_state() -> dict[str, Any]:
    return {
        "channels": {}, "segments": {}, "window_versions": {}, "rules": {},
        "derived": {"series": {}, "alignments": {}, "measurements": {}, "jitter": {}},
        "decisions": {}, "invalidations": [], "imports": [], "history": [],
    }


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(state, default=_json_default))


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def validate_baseline(pack: dict[str, Any]) -> None:
    required = ["case_pack_id", "channels", "segments", "measurement_window_versions", "alignment_rules"]
    missing = [name for name in required if name not in pack]
    if missing:
        raise ImportErrorRollback(f"baseline missing required fields: {','.join(missing)}")
    for segment in pack["segments"]:
        segment_id = segment["segment_id"]
        if "raw_samples" in segment:
            raise ImportErrorRollback("raw samples must be nested by channel")
        for channel_id, capture in segment.get("channels", {}).items():
            if channel_id not in pack["channels"]:
                raise ImportErrorRollback(f"segment {segment_id} has undeclared channel {channel_id}")
            if not isinstance(capture.get("raw_samples"), list):
                raise ImportErrorRollback(f"segment {segment_id}/{channel_id} has no raw sample list")
    for window_id, versions in pack["measurement_window_versions"].items():
        if not isinstance(versions, list) or not versions:
            raise ImportErrorRollback(f"window {window_id} must have at least one version")


def active_version(versions: dict[str, list[dict[str, Any]]], window_id: str) -> dict[str, Any]:
    active = [item for item in versions[window_id] if item.get("status", "active") == "active"]
    return active[-1] if active else versions[window_id][-1]


def build_derived(state: dict[str, Any], rule: dict[str, Any] | None = None,
                  window: dict[str, Any] | None = None,
                  only_channels: set[str] | None = None) -> None:
    if not state["segments"]:
        return
    rule = rule or next(iter(state["rules"].values()))
    window = window or active_version(state["window_versions"], next(iter(state["window_versions"])))
    ref_segment_id = rule["reference_segment"]
    ref_channel_id = rule["reference_channel"]
    threshold = float(rule.get("candidate_threshold", 0.75))
    all_diagnostics: list[dict[str, Any]] = []
    series = state["derived"]["series"]
    for segment_id, segment in state["segments"].items():
        for channel_id in segment["channels"]:
            key = D.stable_id("series", segment_id, channel_id)
            if only_channels and channel_id not in only_channels:
                continue
            built = D.build_channel_series(segment, channel_id, state["channels"][channel_id])
            for diag in built["diagnostics"]:
                diag.update({"segment_id": segment_id, "channel_id": channel_id,
                             "display_id": D.stable_id("diag", segment_id, channel_id, diag.get("code", "X"))})
            series[key] = {
                "id": key,
                "segment_id": segment_id,
                "channel_id": channel_id,
                "time_seconds": built["time_seconds"].tolist(),
                "volts": [None if value != value else float(value) for value in built["volts"]],
                "missing": built["missing"].tolist(),
                "saturated": built["saturated"].tolist(),
                "diagnostics": built["diagnostics"],
                "dependencies": sorted(node.key() for node in
                                       D.segment_nodes(segment_id) | D.channel_nodes(channel_id, segment_id)),
            }
            all_diagnostics.extend(built["diagnostics"])
    reference = D.build_channel_series(state["segments"][ref_segment_id], ref_channel_id,
                                       state["channels"][ref_channel_id])
    alignments = state["derived"]["alignments"]
    measurements = state["derived"]["measurements"]
    jitter = state["derived"]["jitter"]
    for segment_id, segment in state["segments"].items():
        target = D.build_channel_series(segment, ref_channel_id, state["channels"][ref_channel_id])
        candidates = D.alignment_candidates(reference, target, threshold) if segment_id != ref_segment_id else [{
            "candidate_id": "cand:lag:0s", "shift_seconds": 0.0, "score": 1.0,
            "pair_count": int(np_count_valid(target)), "rank": 1, "status": "reference_zero",
        }]
        existing = alignments.get(D.stable_id("align", segment_id, ref_channel_id), {})
        accepted_id = existing.get("accepted_candidate_id")
        if accepted_id and not any(item["candidate_id"] == accepted_id for item in candidates):
            accepted_id = None
        align_id = D.stable_id("align", segment_id, ref_channel_id)
        alignments[align_id] = {
            "id": align_id, "segment_id": segment_id, "channel_id": ref_channel_id,
            "reference_segment_id": ref_segment_id, "candidates": candidates,
            "accepted_candidate_id": accepted_id,
            "ambiguous": len(candidates) > 1 and accepted_id is None,
            "dependencies": sorted(node.key() for node in
                                   D.segment_nodes(segment_id) | D.segment_nodes(ref_segment_id)
                                   | D.channel_nodes(ref_channel_id, segment_id)
                                   | D.channel_nodes(ref_channel_id, ref_segment_id)
                                   | D.rule_nodes(rule["rule_id"])),
        }
        chosen = next((item for item in candidates if item["candidate_id"] == accepted_id), None)
        if chosen is None and len(candidates) == 1:
            chosen = candidates[0]
        trigger_s, trigger_code = D.parse_seconds(segment["trigger"]["timestamp"])
        trigger_diag = []
        if trigger_s == 0:
            trigger_diag.append({"code": "TRIGGER_AT_CAPTURE_BOUNDARY", "severity": "warning",
                                 "display_id": D.stable_id("diag", segment_id, "trigger-boundary")})
        aligned_trigger = None if trigger_s is None or chosen is None else trigger_s - chosen["shift_seconds"]
        jitter_id = D.stable_id("jitter", segment_id)
        jitter[jitter_id] = {
            "id": jitter_id, "segment_id": segment_id,
            "aligned_trigger_seconds": aligned_trigger,
            "reference_trigger_seconds": D.parse_seconds(
                state["segments"][ref_segment_id]["trigger"]["timestamp"])[0],
            "jitter_seconds": None if aligned_trigger is None else aligned_trigger -
            D.parse_seconds(state["segments"][ref_segment_id]["trigger"]["timestamp"])[0],
            "diagnostics": trigger_diag,
            "dependencies": alignments[align_id]["dependencies"] +
            [D.DependencyNode("trigger", segment_id).key(),
             D.DependencyNode("trigger", ref_segment_id).key()],
        }
        for channel_id in segment["channels"]:
            channel_series = D.build_channel_series(segment, channel_id, state["channels"][channel_id])
            shifted = dict(channel_series)
            if chosen is not None:
                shifted["time_seconds"] = channel_series["time_seconds"] - chosen["shift_seconds"]
            result = D.measurement_in_window(shifted, window)
            # Keep reviewers from treating a window that crosses a masked interval as clean.
            inside = (shifted["time_seconds"] >= D.parse_seconds(window["start"])[0]) & \
                     (shifted["time_seconds"] <= D.parse_seconds(window["end"])[0])
            if np.count_nonzero(inside & channel_series["saturated"]):
                result["status"] = "EXCLUDED_SATURATED_WINDOW"
            meas_id = D.stable_id("meas", segment_id, channel_id, window["window_id"],
                                  f"v{window['version']}")
            measurements[meas_id] = {
                "id": meas_id, "segment_id": segment_id, "channel_id": channel_id,
                "window_id": window["window_id"], "window_version": window["version"],
                **result,
                "dependencies": sorted(set(alignments[align_id]["dependencies"] +
                                           [node.key() for node in D.channel_nodes(channel_id, segment_id)] +
                                           [node.key() for node in D.window_nodes(window["window_id"])])),
            }
    state["diagnostics"] = all_diagnostics


def np_count_valid(series: dict[str, Any]) -> int:
    return int(sum(value == value for value in series["volts"]))


def apply_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    kind = event["kind"]
    payload = event["payload"]
    if kind in {"case_imported", "source_imported"}:
        pack = payload["sources"]
        state["case_pack_id"] = pack["case_pack_id"]
        state["title"] = pack.get("title")
        state["channels"] = json.loads(json.dumps(pack["channels"]))
        state["segments"] = {item["segment_id"]: item for item in json.loads(json.dumps(pack["segments"]))}
        state["window_versions"] = json.loads(json.dumps(pack["measurement_window_versions"]))
        state["rules"] = {item["rule_id"]: json.loads(json.dumps(item))
                          for item in pack.get("alignment_rules", {}).values()}
        state["imports"].append({"event_id": event["event_id"], "case_pack_id": pack["case_pack_id"]})
        build_derived(state)
    elif kind == "parameter_revision":
        rule_id = payload["rule_id"]
        before = json.loads(json.dumps(state["rules"][rule_id]))
        state["rules"][rule_id].update(payload.get("patch", {}))
        state["rules"][rule_id]["version"] = int(payload["version"])
        build_derived(state)
        state["invalidations"].append({
            "event_id": event["event_id"], "reason": payload.get("reason"),
            "changed_dependencies": [D.rule_nodes(rule_id).pop().key()],
            "before_rule": before, "scope": "rule_dependents",
        })
    elif kind == "measurement_window_revision":
        window = {"window_id": payload["window_id"], "version": int(payload["version"]),
                  "start": payload["start"], "end": payload["end"],
                  "coordinate_basis": payload.get("coordinate_basis", "aligned_time"),
                  "status": "active", "reason": payload.get("reason")}
        versions = state["window_versions"].setdefault(window["window_id"], [])
        for item in versions:
            item["status"] = "superseded"
        versions.append(window)
        build_derived(state)
        state["invalidations"].append({
            "event_id": event["event_id"], "reason": payload.get("reason"),
            "changed_dependencies": [D.window_nodes(window["window_id"]).pop().key()],
            "scope": "window_dependents",
        })
    elif kind == "window_save":
        window_id = payload["window_id"]
        window = {"window_id": window_id, "version": int(payload["version"]),
                  "start": payload["start"], "end": payload["end"], "status": "active",
                  "coordinate_basis": payload.get("coordinate_basis", "aligned_time")}
        versions = state["window_versions"].setdefault(window_id, [])
        for item in versions:
            item["status"] = "superseded"
        versions.append(window)
        build_derived(state)
    elif kind == "source_data_revision":
        if payload.get("scope") == "local_dependency" and "channel_id" in payload:
            channel_id = payload["channel_id"]
            before = json.loads(json.dumps(state["channels"][channel_id]))
            for name in ["deskew", "vertical_gain", "probe_attenuation", "vertical_offset"]:
                if name in payload:
                    state["channels"][channel_id][name] = payload[name]
            build_derived(state)
            changed = [node.key() for node in D.channel_nodes(channel_id)]
            affected = []
            for group in state["derived"].values():
                if isinstance(group, dict):
                    for object_id, object_value in group.items():
                        if isinstance(object_value, dict) and object_value.get("channel_id") == channel_id:
                            affected.extend(D.affected_scope(object_value.get("dependencies", []), set(changed)))
            state["invalidations"].append({
                "event_id": event["event_id"], "reason": payload.get("reason"),
                "changed_dependencies": changed,
                "affected_dependencies": sorted(set(affected)),
                "before_channel": before, "scope": "local_channel_dependency",
            })
        else:
            raise ValueError("only local source revisions are supported by replay")
    elif kind in {"candidate_decision", "decision_recorded"}:
        segment_id = payload["segment_id"]
        channel_id = payload.get("channel_id", "CH1")
        align_id = D.stable_id("align", segment_id, channel_id)
        if payload.get("decision") == "accept":
            state["decisions"][align_id] = {
                "event_id": event["event_id"], "branch_id": event["branch_id"],
                "candidate_id": payload["candidate_id"], "decision": "accepted",
                "reason": payload.get("reason"),
            }
            state["derived"]["alignments"][align_id]["accepted_candidate_id"] = payload["candidate_id"]
        elif payload.get("decision") in {"reject", "negate"}:
            state["decisions"][D.stable_id("decision", segment_id, payload["candidate_id"])] = {
                "event_id": event["event_id"], "branch_id": event["branch_id"],
                "candidate_id": payload["candidate_id"], "decision": "rejected",
                "reason": payload.get("reason"),
            }
        build_derived(state)
    elif kind in {"counterexample_judgment", "branch_forked", "decision_reverted"}:
        return state
    else:
        raise ValueError(f"unknown event kind {kind}")
    state["history"].append({"event_id": event["event_id"], "ord": event["ord"], "kind": kind})
    return state


def branch_chain(conn: sqlite3.Connection, branch_id: str) -> list[sqlite3.Row]:
    chain = []
    current = branch_id
    while current:
        row = conn.execute("SELECT * FROM branches WHERE branch_id=?", (current,)).fetchone()
        if row is None:
            raise KeyError(f"unknown branch {current}")
        chain.append(row)
        current = row["parent_branch_id"]
    return list(reversed(chain))


def branch_event_rows(conn: sqlite3.Connection, branch_id: str) -> list[sqlite3.Row]:
    selected: dict[int, sqlite3.Row] = {}
    for row in branch_chain(conn, branch_id):
        if row["parent_event_ord"] is None:
            limit = conn.execute("SELECT MAX(ord) FROM events").fetchone()[0] or 0
        else:
            limit = int(row["parent_event_ord"])
        rows = conn.execute("SELECT * FROM events WHERE branch_id=? AND ord<=? ORDER BY ord",
                            (row["branch_id"], limit)).fetchall()
        for event_row in rows:
            selected[int(event_row["ord"])] = event_row
    return [selected[key] for key in sorted(selected)]


def row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    return {"event_id": row["event_id"], "ord": row["ord"], "branch_id": row["branch_id"],
            "kind": row["kind"], "payload": json.loads(row["payload_json"]), "sha256": row["sha256"]}


def state_for_branch(conn: sqlite3.Connection, branch_id: str) -> dict[str, Any]:
    state = initial_state()
    for row in branch_event_rows(conn, branch_id):
        if row["kind"] != "counterexample_judgment":
            apply_event(state, row_to_event(row))
    state["current_branch"] = branch_id
    return state


def _append_judgment_events(conn: sqlite3.Connection, branch_id: str,
                            counterexamples: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    for item in counterexamples:
        judgment = item["machine_judgment"]
        judgment_payload = {
            "counterexample_id": item["counterexample_id"],
            "direct_source_changed": item["direct_source_changed"],
            "changed_segment": item.get("changed_segment"),
            "mutation": item.get("mutation"),
            "adopted_basis": item["adopted_result"]["basis"],
            "not_adopted_basis": item["not_adopted_result"]["basis"],
            **judgment,
        }
        machine = append_event(conn, branch_id, "counterexample_judgment", judgment_payload)
        adopted = append_event(conn, branch_id, "counterexample_judgment", {
            "counterexample_id": item["counterexample_id"], "resolution": "adopted",
            "basis": item["adopted_result"]["basis"], "valid": True,
        })
        rejected = append_event(conn, branch_id, "counterexample_judgment", {
            "counterexample_id": item["counterexample_id"], "resolution": "not_adopted",
            "basis": item["not_adopted_result"]["basis"], "valid": False,
        })
        refs[item["counterexample_id"]] = {
            "machine_judgment_event_id": machine["event_id"],
            "adopted_event_id": adopted["event_id"],
            "not_adopted_event_id": rejected["event_id"],
        }
        conn.execute(
            """INSERT OR REPLACE INTO counterexamples
               (counterexample_id,payload_json,adopted_event_id,not_adopted_event_id,machine_judgment_event_id)
               VALUES(?,?,?,?,?)""",
            (item["counterexample_id"], json.dumps(item, ensure_ascii=False, sort_keys=True),
             adopted["event_id"], rejected["event_id"], machine["event_id"]),
        )
    return refs


def load_case_pack(conn: sqlite3.Connection, case_dir: str | Path = CASE_DIR,
                   replay: bool = False) -> dict[str, Any]:
    init_db(conn)
    if conn.execute("SELECT 1 FROM events LIMIT 1").fetchone():
        return {"status": "already_loaded", "case_dir": str(case_dir)}
    case_dir = Path(case_dir)
    pack = read_json(case_dir / "01_initial_import.json")
    counterexamples = read_json(case_dir / "counterexamples.json")
    replay_events = read_json(case_dir / "02_replay_events.json")
    validate_baseline(pack)
    try:
        ensure_branch(conn)
        files = [case_dir / "01_initial_import.json", case_dir / "counterexamples.json",
                 case_dir / "02_replay_events.json", case_dir / "manifest.json"]
        event = append_event(
            conn, "MAIN", "case_imported",
            {"case_pack_id": pack["case_pack_id"], "title": pack["title"],
             "source_representation": "original JSON with units and coordinate basis retained",
             "sources": pack},
            blobs=[(path, "original_case_byte") for path in files],
        )
        refs = _append_judgment_events(conn, "MAIN", counterexamples)
        result = {"status": "loaded", "event_id": event["event_id"], "counterexample_refs": refs}
        if replay:
            result["replay"] = apply_replay_events(conn, replay_events)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def latest_event_ord(conn: sqlite3.Connection, branch_id: str) -> int:
    row = conn.execute("SELECT MAX(ord) FROM events WHERE branch_id=?", (branch_id,)).fetchone()
    return int(row[0] or 0)


def apply_replay_events(conn: sqlite3.Connection, replay_events: list[dict[str, Any]]) -> dict[str, Any]:
    created: list[dict[str, Any]] = []
    main_before = latest_event_ord(conn, "MAIN")
    branch_after: str | None = None
    for item in sorted(replay_events, key=lambda x: int(x["event_index"])):
        if item["kind"] == "parameter_revision":
            created.append(append_event(conn, "MAIN", "parameter_revision", {
                "rule_id": item["rule_id"], "version": item["version"],
                "patch": item["patch"], "reason": item["reason"],
            }))
        elif item["kind"] == "measurement_window_revision":
            created.append(append_event(conn, "MAIN", "measurement_window_revision", {
                "window_id": item["window_id"], "version": item["version"],
                "start": item["start"], "end": item["end"], "reason": item["reason"],
                "coordinate_basis": "aligned_time",
            }))
        elif item["kind"] == "source_data_revision":
            created.append(append_event(conn, "MAIN", "source_data_revision", item))
        elif item["kind"] == "conflicting_manual_decisions":
            parent_ord = latest_event_ord(conn, "MAIN")
            branches = []
            for decision in item["decisions"]:
                branch_id = decision["branch_id"]
                ensure_branch(conn, branch_id, branch_id, "MAIN", parent_ord)
                event = append_event(conn, branch_id, "candidate_decision", {
                    "segment_id": decision["segment_id"], "channel_id": "CH1",
                    "candidate_id": decision["candidate_id"], "decision": decision["decision"],
                    "reason": decision["reason"], "conflicts_with": [d["branch_id"] for d in item["decisions"]
                                                                    if d["branch_id"] != branch_id],
                })
                branches.append({"branch_id": branch_id, "event_id": event["event_id"]})
            branch_after = item["decisions"][0]["branch_id"]
            created.append({"kind": "conflicting_manual_decisions", "branches": branches})
    return {"events": created, "default_branch": branch_after or "MAIN", "main_base_ord": main_before}


def save_window(conn: sqlite3.Connection, branch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    init_db(conn)
    base_ord = int(payload.get("base_event_ord") or 0)
    current_ord = latest_event_ord(conn, branch_id)
    if base_ord and base_ord != current_ord:
        state = state_for_branch(conn, branch_id)
        window_id = payload["window_id"]
        current = active_version(state["window_versions"], window_id)
        raise ConflictError("window version conflict", {
            "error": "WINDOW_VERSION_CONFLICT",
            "submitted_base_event_ord": base_ord,
            "current_event_ord": current_ord,
            "submitted_window": {"start": payload["start"], "end": payload["end"]},
            "current_window": current,
            "both_differences": {
                "submitted_start": payload["start"], "current_start": current["start"],
                "submitted_end": payload["end"], "current_end": current["end"],
            },
        })
    versions = state_for_branch(conn, branch_id)["window_versions"]
    window_id = payload["window_id"]
    next_version = max(item["version"] for item in versions[window_id]) + 1
    event = append_event(conn, branch_id, "window_save", {
        "window_id": window_id, "version": next_version,
        "start": payload["start"], "end": payload["end"],
        "coordinate_basis": payload.get("coordinate_basis", "aligned_time"),
        "reason": payload.get("reason", "reviewer manual window save"),
    })
    conn.commit()
    return event


def create_decision_branch(conn: sqlite3.Connection, branch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    init_db(conn)
    parent_ord = latest_event_ord(conn, branch_id)
    decision_kind = str(payload["decision"]).lower()
    suffix = "accept" if decision_kind == "accept" else "negate"
    new_branch = payload.get("branch_id") or D.stable_id(
        "BR", branch_id, payload["segment_id"], payload["candidate_id"].split(":")[-1], suffix)
    ensure_branch(conn, new_branch, payload.get("label", new_branch), branch_id, parent_ord)
    event = append_event(conn, new_branch, "candidate_decision", {
        "segment_id": payload["segment_id"],
        "channel_id": payload.get("channel_id", "CH1"),
        "candidate_id": payload["candidate_id"], "decision": decision_kind,
        "reason": payload.get("reason", "manual review decision"),
        "previous_branch_id": branch_id, "previous_event_ord": parent_ord,
    })
    conn.commit()
    return {"branch_id": new_branch, "parent_branch_id": branch_id,
            "parent_event_ord": parent_ord, "event": event}


def restore_event_branch(conn: sqlite3.Connection, event_id: str, label: str | None = None) -> dict[str, Any]:
    init_db(conn)
    row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
    if row is None:
        raise KeyError(event_id)
    parent_ord = int(row["ord"])
    branch_id = D.stable_id("BR-restore", f"{parent_ord:05d}")
    ensure_branch(conn, branch_id, label or f"restore event {parent_ord}", row["branch_id"], parent_ord)
    conn.commit()
    return {"branch_id": branch_id, "parent_branch_id": row["branch_id"], "parent_event_ord": parent_ord}


def export_archive(conn: sqlite3.Connection) -> dict[str, Any]:
    branches = [dict(row) for row in conn.execute("SELECT * FROM branches ORDER BY rowid")]
    events = []
    for row in conn.execute("SELECT * FROM events ORDER BY ord"):
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        events.append(item)
    blobs = []
    for row in conn.execute("SELECT blob_id,filename,sha256,content FROM raw_blobs ORDER BY blob_id"):
        import base64
        blobs.append({"blob_id": row["blob_id"], "filename": row["filename"],
                      "sha256": row["sha256"], "content_base64": base64.b64encode(row["content"]).decode()})
    links = [dict(row) for row in conn.execute("SELECT * FROM event_blobs ORDER BY event_id,role")]
    counters = [dict(row) for row in conn.execute("SELECT * FROM counterexamples ORDER BY counterexample_id")]
    return {"archive_version": "1.0", "exported_at": now(), "branches": branches,
            "events": events, "raw_blobs": blobs, "event_blobs": links, "counterexamples": counters}


def import_archive(conn: sqlite3.Connection, archive: dict[str, Any]) -> dict[str, Any]:
    init_db(conn)
    if conn.execute("SELECT 1 FROM events LIMIT 1").fetchone():
        raise ImportErrorRollback("archive import requires an empty review database")
    required = ["branches", "events", "raw_blobs", "event_blobs", "counterexamples"]
    if any(name not in archive for name in required):
        raise ImportErrorRollback("archive is missing required tables")
    expected = [row for row in archive["events"]]
    for index, event in enumerate(expected, start=1):
        if int(event["ord"]) != index:
            raise ImportErrorRollback("event order is not contiguous")
    try:
        for branch in archive["branches"]:
            conn.execute(
                "INSERT INTO branches(branch_id,parent_branch_id,label,parent_event_ord,created_at) VALUES(?,?,?,?,?)",
                (branch["branch_id"], branch["parent_branch_id"], branch["label"],
                 branch["parent_event_ord"], branch["created_at"]))
        import base64
        for blob in archive["raw_blobs"]:
            content = base64.b64decode(blob["content_base64"])
            if hashlib.sha256(content).hexdigest() != blob["sha256"]:
                raise ImportErrorRollback(f"raw blob checksum failed: {blob['blob_id']}")
            conn.execute(
                "INSERT INTO raw_blobs(blob_id,filename,content,sha256,imported_at) VALUES(?,?,?,?,?)",
                (blob["blob_id"], blob["filename"], content, blob["sha256"], now()))
        for event in expected:
            payload_json = json.dumps(event["payload"], ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(json.dumps(
                {**event["payload"], "event_ord": event["ord"], "branch_id": event["branch_id"]},
                ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if digest != event["sha256"]:
                raise ImportErrorRollback(f"event checksum failed at ord {event['ord']}")
            conn.execute(
                "INSERT INTO events(event_id,ord,branch_id,kind,payload_json,created_at,sha256) VALUES(?,?,?,?,?,?,?)",
                (event["event_id"], event["ord"], event["branch_id"], event["kind"], payload_json,
                 event["created_at"], event["sha256"]))
        for link in archive["event_blobs"]:
            conn.execute("INSERT INTO event_blobs(event_id,blob_id,role) VALUES(?,?,?)",
                         (link["event_id"], link["blob_id"], link["role"]))
        for item in archive["counterexamples"]:
            conn.execute(
                """INSERT INTO counterexamples
                   (counterexample_id,payload_json,adopted_event_id,not_adopted_event_id,machine_judgment_event_id)
                   VALUES(?,?,?,?,?)""",
                (item["counterexample_id"], item["payload_json"], item["adopted_event_id"],
                 item["not_adopted_event_id"], item["machine_judgment_event_id"]))
        # Replay every imported event to prove the materialized derivation can be reconstructed.
        state_for_branch(conn, "MAIN")
        conn.commit()
        return {"status": "imported", "events": len(expected),
                "branches": len(archive["branches"]),
                "raw_blobs": len(archive["raw_blobs"])}
    except Exception:
        conn.rollback()
        raise


def list_branches(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM branches ORDER BY created_at,branch_id").fetchall()
    result = []
    for row in rows:
        state = state_for_branch(conn, row["branch_id"]) if conn.execute(
            "SELECT 1 FROM events LIMIT 1").fetchone() else initial_state()
        result.append({
            "branch_id": row["branch_id"], "label": row["label"],
            "parent_branch_id": row["parent_branch_id"], "parent_event_ord": row["parent_event_ord"],
            "tip_event_ord": latest_event_ord(conn, row["branch_id"]),
            "alignment_acceptance": {key: value.get("accepted_candidate_id")
                                      for key, value in state["derived"]["alignments"].items()},
        })
    return result


def invalidate_derived_by_dependencies(state: dict[str, Any], changed: set[str]) -> dict[str, list[str]]:
    affected: dict[str, list[str]] = {}
    for name, group in state["derived"].items():
        if not isinstance(group, dict):
            continue
        hit = []
        for object_id, item in group.items():
            if isinstance(item, dict) and D.affected_scope(item.get("dependencies", []), changed):
                hit.append(object_id)
        if hit:
            affected[name] = sorted(hit)
    return affected
