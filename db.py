"""SQLite persistence for the waveform overlay review tool.

Schema is versioned via PRAGMA user_version and migrated stepwise so that
state-migration tests can build an old database and upgrade it in place.
"""
import json
import sqlite3
import time
import uuid

SCHEMA_VERSION = 2

DDL_V1 = """
CREATE TABLE meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE imports(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE TABLE segments(
  id TEXT PRIMARY KEY,
  import_id TEXT NOT NULL REFERENCES imports(id),
  seg_key TEXT NOT NULL,
  channel TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  raw_json TEXT NOT NULL,
  source_hash TEXT NOT NULL
);
CREATE TABLE branches(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  parent_id TEXT,
  base_event_id TEXT,
  head_event_id TEXT,
  created_seq INTEGER NOT NULL
);
CREATE TABLE events(
  id TEXT PRIMARY KEY,
  branch_id TEXT NOT NULL REFERENCES branches(id),
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  prev_id TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE derived(
  id TEXT PRIMARY KEY,
  branch_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  deps TEXT NOT NULL,
  payload TEXT NOT NULL,
  valid INTEGER NOT NULL DEFAULT 1,
  computed_at TEXT NOT NULL
);
"""

# v2: import-time unit diagnostics are persisted alongside the raw segment so
# that unconvertible values keep a machine-readable reason next to the raw
# bytes (they are never silently normalized away).
MIGRATIONS = {
    2: "ALTER TABLE segments ADD COLUMN diagnostics TEXT NOT NULL DEFAULT '[]';",
}

TABLES = ["meta", "imports", "segments", "branches", "events", "derived"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def new_id(prefix):
    return "%s-%s" % (prefix, uuid.uuid4().hex[:12])


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def migrate(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript(DDL_V1)
        version = 1
    while version < SCHEMA_VERSION:
        version += 1
        conn.execute(MIGRATIONS[version])
    conn.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
    conn.commit()


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_meta(conn, key, value):
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )


def list_events(conn, branch_id):
    rows = conn.execute(
        "SELECT * FROM events WHERE branch_id=? ORDER BY seq", (branch_id,)
    ).fetchall()
    return [dict(r, payload=json.loads(r["payload"])) for r in rows]


def append_event(conn, branch_id, kind, payload, expected_head="__check__"):
    """Append an event; enforces optimistic concurrency on the branch head."""
    br = conn.execute("SELECT * FROM branches WHERE id=?", (branch_id,)).fetchone()
    if br is None:
        raise KeyError("branch not found: %s" % branch_id)
    head = br["head_event_id"]
    if expected_head != "__check__" and expected_head != head:
        later = []
        if head:
            evs = list_events(conn, branch_id)
            seen = expected_head is None
            for e in evs:
                if seen:
                    later.append(e)
                elif e["id"] == expected_head:
                    seen = True
        raise ConflictError(head, later)
    seq = conn.execute(
        "SELECT COALESCE(MAX(seq),0)+1 AS s FROM events WHERE branch_id=?",
        (branch_id,),
    ).fetchone()["s"]
    eid = new_id("evt")
    conn.execute(
        "INSERT INTO events(id,branch_id,seq,kind,payload,prev_id,created_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (eid, branch_id, seq, kind, json.dumps(payload), head, now()),
    )
    conn.execute("UPDATE branches SET head_event_id=? WHERE id=?", (eid, branch_id))
    return eid


class ConflictError(Exception):
    def __init__(self, current_head, their_events):
        self.current_head = current_head
        self.their_events = their_events
        super().__init__("branch head conflict")
