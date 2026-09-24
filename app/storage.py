"""SQLite storage: immutable raw blobs, source facts, event log per branch,
findings (machine-readable judgement events), and the dependency graph used
for scoped invalidation. Nothing here rewrites history."""
from __future__ import annotations

import json
import sqlite3
import time

SCHEMA_VERSION = 2

_SCHEMA_V1 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE blobs (hash TEXT PRIMARY KEY, content BLOB NOT NULL);
CREATE TABLE facts (
    fact_id TEXT PRIMARY KEY,
    fact_type TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    blob_hash TEXT,
    import_event_id TEXT,
    created_seq INTEGER
);
CREATE TABLE branches (
    branch_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_branch_id TEXT,
    fork_event_seq INTEGER,
    head_seq INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE events (
    branch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (branch_id, seq)
);
"""

_SCHEMA_V2_DELTA = """
CREATE TABLE findings (
    branch_id TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    code TEXT NOT NULL,
    adopted INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (branch_id, finding_id)
);
CREATE TABLE graph_edges (
    branch_id TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    kind TEXT NOT NULL,
    PRIMARY KEY (branch_id, state_hash, src, dst, kind)
);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["value"]) if row else 0


def init_db(conn: sqlite3.Connection) -> None:
    """Create the schema or migrate older snapshots forward."""
    version = _schema_version(conn)
    if version == 0:
        conn.executescript(_SCHEMA_V1)
        conn.executescript(_SCHEMA_V2_DELTA)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),))
        conn.commit()
        return
    if version == 1:
        # Migration v1 -> v2: findings and dependency graph tables.
        conn.executescript(_SCHEMA_V2_DELTA)
        conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'",
            (str(SCHEMA_VERSION),))
        conn.commit()
        return
    if version != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported schema version {version}")


def put_blob(conn: sqlite3.Connection, blob_hash: str, content: bytes) -> None:
    conn.execute("INSERT OR IGNORE INTO blobs(hash, content) VALUES(?, ?)",
                 (blob_hash, content))


def get_blob(conn: sqlite3.Connection, blob_hash: str) -> bytes | None:
    row = conn.execute("SELECT content FROM blobs WHERE hash=?",
                       (blob_hash,)).fetchone()
    return bytes(row["content"]) if row else None


def insert_fact(conn, fact_id, fact_type, fact_key, version, payload,
                blob_hash, import_event_id, created_seq):
    conn.execute(
        "INSERT INTO facts(fact_id, fact_type, fact_key, version, payload_json,"
        " blob_hash, import_event_id, created_seq) VALUES(?,?,?,?,?,?,?,?)",
        (fact_id, fact_type, fact_key, version, json.dumps(payload),
         blob_hash, import_event_id, created_seq))


def facts(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM facts ORDER BY fact_key, version").fetchall()
    return [dict(r) for r in rows]


def latest_facts(conn) -> dict[str, dict]:
    """fact_key -> latest-version fact row."""
    out: dict[str, dict] = {}
    for row in conn.execute("SELECT * FROM facts ORDER BY version"):
        out[row["fact_key"]] = dict(row)
    return out


def create_branch(conn, branch_id, name, parent_branch_id, fork_event_seq,
                  head_seq=0):
    conn.execute(
        "INSERT INTO branches(branch_id, name, parent_branch_id,"
        " fork_event_seq, head_seq) VALUES(?,?,?,?,?)",
        (branch_id, name, parent_branch_id, fork_event_seq, head_seq))


def branches(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM branches ORDER BY rowid")]


def get_branch(conn, branch_id) -> dict | None:
    row = conn.execute("SELECT * FROM branches WHERE branch_id=?",
                       (branch_id,)).fetchone()
    return dict(row) if row else None


def append_event(conn, branch_id, event_id, event_type, payload,
                 expected_head=None):
    """Append with optimistic concurrency: ``expected_head`` must match the
    current head sequence or a ConflictError is raised."""
    branch = get_branch(conn, branch_id)
    if branch is None:
        raise KeyError(f"unknown branch {branch_id}")
    if expected_head is not None and branch["head_seq"] != expected_head:
        raise ConflictError(branch_id, expected_head, branch["head_seq"])
    seq = branch["head_seq"] + 1
    conn.execute(
        "INSERT INTO events(branch_id, seq, event_id, event_type, payload_json,"
        " created_at) VALUES(?,?,?,?,?,?)",
        (branch_id, seq, event_id, event_type, json.dumps(payload),
         time.time()))
    conn.execute("UPDATE branches SET head_seq=? WHERE branch_id=?",
                 (seq, branch_id))
    return seq


class ConflictError(Exception):
    def __init__(self, branch_id, expected, actual):
        super().__init__(
            f"branch {branch_id} head moved: expected {expected},"
            f" actual {actual}")
        self.branch_id = branch_id
        self.expected = expected
        self.actual = actual


def events(conn, branch_id, upto_seq=None) -> list[dict]:
    if upto_seq is None:
        rows = conn.execute(
            "SELECT * FROM events WHERE branch_id=? ORDER BY seq",
            (branch_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM events WHERE branch_id=? AND seq<=? ORDER BY seq",
            (branch_id, upto_seq)).fetchall()
    return [dict(r) for r in rows]


def put_findings(conn, branch_id, state_hash, items):
    for item in items:
        conn.execute(
            "INSERT OR REPLACE INTO findings(branch_id, state_hash, finding_id,"
            " code, adopted, payload_json) VALUES(?,?,?,?,?,?)",
            (branch_id, state_hash, item["finding_id"], item["code"],
             1 if item.get("adopted") else 0, json.dumps(item)))


def findings_for(conn, branch_id, state_hash=None) -> list[dict]:
    if state_hash is None:
        rows = conn.execute(
            "SELECT payload_json FROM findings WHERE branch_id=?",
            (branch_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT payload_json FROM findings WHERE branch_id=?"
            " AND state_hash=?", (branch_id, state_hash)).fetchall()
    return [json.loads(r["payload_json"]) for r in rows]


def put_edges(conn, branch_id, state_hash, edges):
    for src, dst, kind in edges:
        conn.execute(
            "INSERT OR IGNORE INTO graph_edges(branch_id, state_hash, src, dst,"
            " kind) VALUES(?,?,?,?,?)",
            (branch_id, state_hash, src, dst, kind))


def edges_for(conn, branch_id, state_hash) -> list[tuple]:
    rows = conn.execute(
        "SELECT src, dst, kind FROM graph_edges WHERE branch_id=?"
        " AND state_hash=?", (branch_id, state_hash)).fetchall()
    return [(r["src"], r["dst"], r["kind"]) for r in rows]


def export_bundle(conn) -> dict:
    return {
        "format": "waveform-overlay-bundle",
        "schema_version": SCHEMA_VERSION,
        "meta": {r["key"]: r["value"] for r in
                 conn.execute("SELECT key, value FROM meta")},
        "blobs": [{"hash": r["hash"],
                   "content_b64": _b64(bytes(r["content"]))}
                  for r in conn.execute("SELECT * FROM blobs ORDER BY hash")],
        "facts": facts(conn),
        "branches": branches(conn),
        "events": [dict(r) for r in conn.execute(
            "SELECT * FROM events ORDER BY branch_id, seq")],
        "findings": [dict(r) for r in conn.execute(
            "SELECT * FROM findings ORDER BY branch_id, finding_id")],
        "graph_edges": [dict(r) for r in conn.execute(
            "SELECT * FROM graph_edges ORDER BY branch_id, state_hash")],
    }


def import_bundle(conn, bundle: dict) -> None:
    """Import a full bundle into an empty database. All-or-nothing: the caller
    wraps this in a transaction so a failure rolls back the whole change."""
    if bundle.get("format") != "waveform-overlay-bundle":
        raise ValueError("not a waveform-overlay bundle")
    if int(bundle.get("schema_version", 0)) > SCHEMA_VERSION:
        raise ValueError("bundle schema newer than this build")
    for blob in bundle.get("blobs", []):
        put_blob(conn, blob["hash"], _unb64(blob["content_b64"]))
    for fact in bundle.get("facts", []):
        conn.execute(
            "INSERT INTO facts(fact_id, fact_type, fact_key, version,"
            " payload_json, blob_hash, import_event_id, created_seq)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (fact["fact_id"], fact["fact_type"], fact["fact_key"],
             fact["version"], fact["payload_json"], fact["blob_hash"],
             fact["import_event_id"], fact["created_seq"]))
    for branch in bundle.get("branches", []):
        conn.execute(
            "INSERT INTO branches(branch_id, name, parent_branch_id,"
            " fork_event_seq, head_seq) VALUES(?,?,?,?,?)",
            (branch["branch_id"], branch["name"], branch["parent_branch_id"],
             branch["fork_event_seq"], branch["head_seq"]))
    for event in bundle.get("events", []):
        conn.execute(
            "INSERT INTO events(branch_id, seq, event_id, event_type,"
            " payload_json, created_at) VALUES(?,?,?,?,?,?)",
            (event["branch_id"], event["seq"], event["event_id"],
             event["event_type"], event["payload_json"], event["created_at"]))
    for finding in bundle.get("findings", []):
        conn.execute(
            "INSERT INTO findings(branch_id, state_hash, finding_id, code,"
            " adopted, payload_json) VALUES(?,?,?,?,?,?)",
            (finding["branch_id"], finding["state_hash"],
             finding["finding_id"], finding["code"], finding["adopted"],
             finding["payload_json"]))
    for edge in bundle.get("graph_edges", []):
        conn.execute(
            "INSERT INTO graph_edges(branch_id, state_hash, src, dst, kind)"
            " VALUES(?,?,?,?,?)",
            (edge["branch_id"], edge["state_hash"], edge["src"], edge["dst"],
             edge["kind"]))


def _b64(content: bytes) -> str:
    import base64
    return base64.b64encode(content).decode("ascii")


def _unb64(text: str) -> bytes:
    import base64
    return base64.b64decode(text.encode("ascii"))
