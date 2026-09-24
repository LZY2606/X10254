from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 2

MIGRATION_1 = """
PRAGMA foreign_keys = ON;
CREATE TABLE import_batches (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    archive_kind TEXT NOT NULL,
    raw_bytes BLOB NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE materials (
    id TEXT PRIMARY KEY,
    batch_id TEXT REFERENCES import_batches(id),
    introduced_event_id TEXT,
    supersedes_material_id TEXT REFERENCES materials(id),
    stable_key TEXT NOT NULL,
    material_type TEXT NOT NULL,
    segment_id TEXT,
    channel_id TEXT,
    ordinal INTEGER NOT NULL,
    raw_bytes BLOB NOT NULL,
    raw_json TEXT NOT NULL,
    sha256 TEXT NOT NULL
);
CREATE TABLE branches (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    parent_branch_id TEXT REFERENCES branches(id),
    fork_event_id TEXT,
    head_event_id TEXT,
    label TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    branch_id TEXT NOT NULL REFERENCES branches(id),
    parent_event_id TEXT REFERENCES events(id),
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    actor TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    parameter_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(branch_id, seq)
);
CREATE TABLE snapshots (
    event_id TEXT PRIMARY KEY REFERENCES events(id),
    branch_id TEXT NOT NULL REFERENCES branches(id),
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE judgments (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(id),
    code TEXT NOT NULL,
    severity TEXT NOT NULL,
    target_id TEXT NOT NULL,
    basis_json TEXT NOT NULL
);
CREATE INDEX idx_materials_stable_key ON materials(stable_key);
CREATE INDEX idx_events_branch ON events(branch_id, seq);
CREATE INDEX idx_judgments_event ON judgments(event_id);
"""

MIGRATION_2 = """
ALTER TABLE materials ADD COLUMN content_hash TEXT;
UPDATE materials SET content_hash = sha256;
CREATE INDEX idx_materials_content_hash ON materials(content_hash);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or os.environ.get("WAVE_REVIEW_DB", "waveform_review.sqlite3")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def run_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(f"database schema {current} is newer than {SCHEMA_VERSION}")
    for version in range(current + 1, SCHEMA_VERSION + 1):
 sql = MIGRATION_1 if version == 1 else MIGRATION_2
        conn.executescript(sql)
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def material_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    raw = json.loads(row["raw_json"])
    return {
        "id": row["id"],
        "batch_id": row["batch_id"],
        "introduced_event_id": row["introduced_event_id"],
        "supersedes_material_id": row["supersedes_material_id"],
        "stable_key": row["stable_key"],
        "material_type": row["material_type"],
        "segment_id": row["segment_id"],
        "channel_id": row["channel_id"],
        "ordinal": row["ordinal"],
        "raw": raw,
        "raw_bytes_base64": base64.b64encode(row["raw_bytes"]).decode("ascii"),
        "sha256": row["sha256"],
        "content_hash": row["content_hash"],
    }


def export_archive(conn: sqlite3.Connection) -> dict[str, Any]:
    batches = [dict(r) for r in conn.execute("SELECT * FROM import_batches ORDER BY created_at, id")]
    materials = [material_row_to_dict(r) for r in conn.execute("SELECT * FROM materials ORDER BY ordinal, id")]
    branches = [dict(r) for r in conn.execute("SELECT * FROM branches ORDER BY created_at, id")]
    events_rows = conn.execute("SELECT * FROM events ORDER BY branch_id, seq").fetchall()
    snapshots = [
        {
            "event_id": r["event_id"],
            "branch_id": r["branch_id"],
            "snapshot": json.loads(r["snapshot_json"]),
            "created_at": r["created_at"],
        }
        for r in conn.execute("SELECT * FROM snapshots ORDER BY event_id")
    ]
    judgments = [dict(r) for r in conn.execute("SELECT * FROM judgments ORDER BY event_id, id")]
    for batch in batches:
        batch["raw_bytes_base64"] = base64.b64encode(batch.pop("raw_bytes")).decode("ascii")
    for judgment in judgments:
        judgment["basis"] = json.loads(judgment.pop("basis_json"))
    events = []
    for row in events_rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        events.append(item)
    return {
        "archive_version": 1,
        "exported_at": utc_now(),
        "title": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "batches": batches,
        "materials": materials,
        "branches": branches,
        "events": events,
        "snapshots": snapshots,
        "judgments": judgments,
    }


def _decode_raw(material: dict[str, Any]) -> bytes:
    if material.get("raw_bytes_base64"):
        return base64.b64decode(material["raw_bytes_base64"])
    return canonical_bytes(material["raw"])


def _validate_archive(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("archive must be a JSON object")
    for key in ("materials", "branches", "events", "snapshots", "judgments"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"archive field {key} must be a list")


def import_archive(conn: sqlite3.Connection, payload: dict[str, Any], raw_archive: bytes) -> None:
    _validate_archive(payload)
    try:
        with conn:
            conn.execute("DELETE FROM judgments")
            conn.execute("DELETE FROM snapshots")
            conn.execute("DELETE FROM events")
            conn.execute("DELETE FROM branches")
            conn.execute("DELETE FROM materials")
            conn.execute("DELETE FROM import_batches")
            for batch in payload["batches"]:
                raw = base64.b64decode(batch["raw_bytes_base64"])
                conn.execute(
                    """INSERT INTO import_batches
                    (id, case_id, archive_kind, raw_bytes, sha256, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (batch["id"], batch["case_id"], batch.get("archive_kind", "case_pack"),
                     raw, batch["sha256"], batch["created_at"]),
                )
            for material in sorted(payload["materials"], key=lambda x: (x["ordinal"], x["id"])):
                raw = _decode_raw(material)
                raw_json = json.dumps(material["raw"], ensure_ascii=False, sort_keys=True)
                conn.execute(
                    """INSERT INTO materials
                    (id, batch_id, introduced_event_id, supersedes_material_id, stable_key,
                     material_type, segment_id, channel_id, ordinal, raw_bytes, raw_json, sha256, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (material["id"], material.get("batch_id"), material.get("introduced_event_id"),
                     material.get("supersedes_material_id"), material["stable_key"],
                     material["material_type"], material.get("segment_id"), material.get("channel_id"),
                     material["ordinal"], raw, raw_json, material["sha256"],
                     material.get("content_hash", material["sha256"])),
                )
            for branch in payload["branches"]:
                conn.execute(
                    """INSERT INTO branches
                    (id, case_id, parent_branch_id, fork_event_id, head_event_id, label, kind, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (branch["id"], branch["case_id"], branch.get("parent_branch_id"),
                     branch.get("fork_event_id"), branch.get("head_event_id"), branch["label"],
                     branch["kind"], branch["created_at"], branch["status"]),
                )
            for event in payload["events"]:
                conn.execute(
                    """INSERT INTO events
                    (id, branch_id, parent_event_id, seq, type, actor, message, created_at,
                     rule_version, parameter_version, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (event["id"], event["branch_id"], event.get("parent_event_id"), event["seq"],
                     event["type"], event["actor"], event["message"], event["created_at"],
                     event["rule_version"], event["parameter_version"],
                     json.dumps(event["payload"], ensure_ascii=False, sort_keys=True)),
                )
            for snapshot in payload["snapshots"]:
                conn.execute(
                    "INSERT INTO snapshots (event_id, branch_id, snapshot_json, created_at) VALUES (?, ?, ?, ?)",
                    (snapshot["event_id"], snapshot["branch_id"],
                     json.dumps(snapshot["snapshot"], ensure_ascii=False, sort_keys=True),
                     snapshot["created_at"]),
                )
            for judgment in payload["judgments"]:
                conn.execute(
                    "INSERT INTO judgments (id, event_id, code, severity, target_id, basis_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (judgment["id"], judgment["event_id"], judgment["code"], judgment["severity"],
                     judgment["target_id"], json.dumps(judgment["basis"], ensure_ascii=False, sort_keys=True)),
                )
    except Exception:
        conn.rollback()
        raise


def has_case(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT EXISTS(SELECT 1 FROM branches)").fetchone()[0] == 1
