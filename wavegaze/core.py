"""稳定身份 + 事件日志原语."""
from __future__ import annotations

import hashlib
import json
import sqlite3

from .db import utcnow


def next_id(conn: sqlite3.Connection, prefix: str, width: int = 4) -> str:
    conn.execute("INSERT OR IGNORE INTO counters(name, value) VALUES (?, 0)", (prefix,))
    conn.execute("UPDATE counters SET value = value + 1 WHERE name = ?", (prefix,))
    value = conn.execute("SELECT value FROM counters WHERE name = ?", (prefix,)).fetchone()[0]
    return f"{prefix}_{value:0{width}d}"


def record_event(conn, kind: str, branch_id: str | None = None, payload: dict | None = None):
    eid = next_id(conn, "evt", 6)
    cur = conn.execute(
        "INSERT INTO events(id, branch_id, kind, payload, created_at) VALUES (?,?,?,?,?)",
        (eid, branch_id, kind, json.dumps(payload or {}, sort_keys=True), utcnow()),
    )
    return eid, cur.lastrowid


def create_branch(conn, parent_id: str | None = None, created_event: str | None = None) -> str:
    bid = next_id(conn, "br")
    conn.execute(
        "INSERT INTO branches(id, parent_id, created_event, version, status, created_at)"
        " VALUES (?,?,?,0,'open',?)",
        (bid, parent_id, created_event, utcnow()),
    )
    return bid


def bump_branch(conn, branch_id: str) -> int:
    conn.execute("UPDATE branches SET version = version + 1 WHERE id = ?", (branch_id,))
    return conn.execute("SELECT version FROM branches WHERE id = ?", (branch_id,)).fetchone()[0]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
