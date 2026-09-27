"""SQLite-backed append-only event store with branches and replay.

Source bytes are immutable ``source_nodes``.  Every interpretation, ruling,
    parameter revision and window revision is an event on a branch.  Replaying
events recreates any historical snapshot; nothing destructive is allowed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_nodes (
    node_id      TEXT NOT NULL,
    branch_id    TEXT NOT NULL,
    kind         TEXT NOT NULL,
    address      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    bytes_json   TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    supersedes   TEXT,
    PRIMARY KEY (node_id, branch_id)
);
CREATE TABLE IF NOT EXISTS branches (
    branch_id    TEXT PRIMARY KEY,
    material_id  TEXT NOT NULL,
    parent_branch TEXT,
    parent_seq   INTEGER,
    title        TEXT NOT NULL,
    created_seq  INTEGER NOT NULL,
    ordinal      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER NOT NULL,
    branch_id    TEXT NOT NULL,
    event_id     TEXT NOT NULL,
    type         TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    hash         TEXT NOT NULL,
    PRIMARY KEY (branch_id, seq)
);
CREATE TABLE IF NOT EXISTS snapshots (
    branch_id    TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (branch_id, seq)
);
CREATE TABLE IF NOT EXISTS archive_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("SCOPEAPP_DB", "scopeapp.db")
        self._lock = threading.RLock()
        self._init()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO archive_meta(key, value) VALUES('schema_version','1')"
            )

    def schema_version(self, conn) -> int:
        row = conn.execute(
            "SELECT value FROM archive_meta WHERE key='schema_version'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def lock(self):
        return self._lock

    def branch_exists(self, branch_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM branches WHERE branch_id=?", (branch_id,)
            ).fetchone()
            return row is not None

    def _next_branch_ordinal(self, conn) -> int:
        row = conn.execute("SELECT COALESCE(MAX(ordinal)+1,0) AS n FROM branches").fetchone()
        return int(row["n"])

    def list_branches(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM branches ORDER BY ordinal"
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                head = conn.execute(
                    "SELECT MAX(seq) AS n FROM events WHERE branch_id=?",
                    (row["branch_id"],),
                ).fetchone()
                item["head_seq"] = int(head["n"] or 0)
                result.append(item)
            return result

    def _read_nodes(self, conn, branch_id: str) -> list[dict]:
        rows = conn.execute(
            "SELECT * FROM source_nodes WHERE branch_id=? ORDER BY ordinal",
            (branch_id,),
        ).fetchall()
        nodes = []
        for row in rows:
            nodes.append({
                "node_id": row["node_id"], "kind": row["kind"],
                "address": row["address"], "content_hash": row["content_hash"],
                "content": json.loads(row["bytes_json"]),
                "ordinal": row["ordinal"], "supersedes": row["supersedes"],
            })
        return nodes

    def read_nodes(self, branch_id: str) -> list[dict]:
        with self.connect() as conn:
            return self._read_nodes(conn, branch_id)

    def write_snapshot(self, branch_id: str, seq: int, snapshot: dict) -> str:
        with self.lock(), self.connect() as conn:
            return self._write_snapshot(conn, branch_id, seq, snapshot)

    def _write_snapshot(self, conn, branch_id: str, seq: int,
                        snapshot: dict) -> str:
        body = canonical_json(snapshot)
        digest = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT OR REPLACE INTO snapshots(branch_id, seq, snapshot_json,"
            " content_hash) VALUES(?,?,?,?)",
            (branch_id, seq, body, digest),
        )
        return digest

    def read_snapshot(self, branch_id: str, seq: int | None = None) -> dict:
        with self.connect() as conn:
            if seq is None:
                row = conn.execute(
                    "SELECT snapshot_json FROM snapshots WHERE branch_id=?"
                    " ORDER BY seq DESC LIMIT 1", (branch_id,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT snapshot_json FROM snapshots WHERE branch_id=? AND seq=?",
                    (branch_id, seq),
                ).fetchone()
            if row is None:
                raise KeyError(f"no snapshot {branch_id}@{seq}")
            return json.loads(row["snapshot_json"])
