"""SQLite persistence: sources, branches, events, derived values, diagnostics.

Source material and interpretation are stored separately. Imports are
transactional: a failed import rolls back completely and leaves no partial
records. Re-interpretation happens on new branches; old branches, human
rulings and the raw bytes they reference stay readable.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sqlite3
from contextlib import contextmanager

SCHEMA_VERSION = 2

# Historical v1 schema, kept so migration tests can build an old database.
SCHEMA_V1 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE imports (id TEXT PRIMARY KEY, seq INTEGER NOT NULL, label TEXT NOT NULL,
  created_at TEXT NOT NULL, status TEXT NOT NULL, error TEXT);
CREATE TABLE sources (id TEXT PRIMARY KEY, import_id TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL,
  raw TEXT NOT NULL, raw_format TEXT NOT NULL, units TEXT NOT NULL,
  basis TEXT NOT NULL, meta TEXT NOT NULL);
CREATE TABLE branches (id TEXT PRIMARY KEY, name TEXT NOT NULL, parent TEXT,
  base_event TEXT, created_at TEXT NOT NULL);
CREATE TABLE events (id TEXT PRIMARY KEY, branch TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE derived (id TEXT PRIMARY KEY, branch TEXT NOT NULL, dkey TEXT NOT NULL,
  version INTEGER NOT NULL, deps TEXT NOT NULL, value TEXT NOT NULL,
  event_id TEXT NOT NULL, created_seq INTEGER NOT NULL, invalidated_seq INTEGER);
"""

MIGRATIONS = {
    1: """
ALTER TABLE events ADD COLUMN undo_of TEXT;
CREATE TABLE diagnostics (id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  severity TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL);
""",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS imports (id TEXT PRIMARY KEY, seq INTEGER NOT NULL, label TEXT NOT NULL,
  created_at TEXT NOT NULL, status TEXT NOT NULL, error TEXT);
CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, import_id TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL,
  raw TEXT NOT NULL, raw_format TEXT NOT NULL, units TEXT NOT NULL,
  basis TEXT NOT NULL, meta TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS branches (id TEXT PRIMARY KEY, name TEXT NOT NULL, parent TEXT,
  base_event TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, branch TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL, undo_of TEXT);
CREATE TABLE IF NOT EXISTS derived (id TEXT PRIMARY KEY, branch TEXT NOT NULL, dkey TEXT NOT NULL,
  version INTEGER NOT NULL, deps TEXT NOT NULL, value TEXT NOT NULL,
  event_id TEXT NOT NULL, created_seq INTEGER NOT NULL, invalidated_seq INTEGER);
CREATE TABLE IF NOT EXISTS diagnostics (id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  severity TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL);
"""

TABLES = ["imports", "sources", "branches", "events", "derived", "diagnostics"]


class Conflict(Exception):
    def __init__(self, branch, base_head, current_head, events_since):
        self.branch = branch
        self.base_head = base_head
        self.current_head = current_head
        self.events_since = events_since
        super().__init__(f"branch {branch!r} head moved: {base_head!r} -> {current_head!r}")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


class Store:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.migrate()

    def close(self):
        self.conn.close()

    # ----- schema -----
    def migrate(self):
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            self.conn.executescript(SCHEMA)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.conn.commit()
            return
        while version < SCHEMA_VERSION:
            self.conn.executescript(MIGRATIONS[version])
            version += 1
            self.conn.execute(f"PRAGMA user_version = {version}")
            self.conn.commit()
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"database schema v{version} newer than supported v{SCHEMA_VERSION}")

    def schema_version(self):
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    # ----- imports (transactional) -----
    def _next_seq(self, table):
        row = self.conn.execute(f"SELECT COALESCE(MAX(seq),0)+1 AS s FROM {table}").fetchone()
        return int(row["s"])

    @contextmanager
    def import_txn(self, label):
        seq = self._next_seq("imports")
        imp_id = f"imp-{seq:03d}"
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "INSERT INTO imports (id, seq, label, created_at, status, error) VALUES (?,?,?,?,?,?)",
                (imp_id, seq, label, _now(), "pending", None))
            yield imp_id
            self.conn.execute("UPDATE imports SET status='committed' WHERE id=?", (imp_id,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def add_source(self, imp_id, kind, name, version, raw, raw_format, units, basis, meta):
        src_id = f"{name}#v{version}"
        self.conn.execute(
            "INSERT INTO sources (id, import_id, seq, kind, name, version, raw, raw_format, units, basis, meta)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (src_id, imp_id, self._next_seq("sources"), kind, name, version, raw, raw_format,
             json.dumps(units, ensure_ascii=False), basis, json.dumps(meta, ensure_ascii=False)))
        return src_id

    def add_diagnostic(self, source_id, severity, message):
        if self.conn.execute(
                "SELECT 1 FROM diagnostics WHERE source_id=? AND message=?",
                (source_id, message)).fetchone():
            return None
        count = self.conn.execute("SELECT COUNT(*) AS c FROM diagnostics").fetchone()["c"]
        diag_id = f"diag-{count + 1:03d}"
        self.conn.execute(
            "INSERT INTO diagnostics (id, source_id, severity, message, created_at) VALUES (?,?,?,?,?)",
            (diag_id, source_id, severity, message, _now()))
        self.conn.commit()
        return diag_id

    def diagnostics(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM diagnostics ORDER BY rowid")]

    # ----- branches & events -----
    def branch_exists(self, name):
        return bool(self.conn.execute("SELECT 1 FROM branches WHERE id=?", (name,)).fetchone())

    def list_branches(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM branches ORDER BY rowid")]

    def create_branch(self, name, parent=None):
        base_event = None
        if parent:
            base_event, _ = self.head(parent)
        self.conn.execute(
            "INSERT INTO branches (id, name, parent, base_event, created_at) VALUES (?,?,?,?,?)",
            (name, name, parent, base_event, _now()))
        if parent:
            for r in self.visible_derived(parent):
                self._insert_derived(name, r["dkey"], json.loads(r["deps"]),
                                     json.loads(r["value"]), base_event or "fork",
                                     created_seq=0, version=r["version"])
            inherited = {s: r for s, r in self.effective_rulings(parent).items()}
            self._add_event(name, "fork",
                            {"subject": name, "parent": parent, "base_event": base_event,
                             "inherited_rulings": inherited})
        self.conn.commit()
        return name

    def _branch_seq(self, branch):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0) AS s FROM events WHERE branch=?", (branch,)).fetchone()
        return int(row["s"])

    def _add_event(self, branch, kind, payload, undo_of=None):
        seq = self._branch_seq(branch) + 1
        eid = f"ev-{branch}-{seq:04d}"
        self.conn.execute(
            "INSERT INTO events (id, branch, seq, kind, payload, created_at, undo_of) VALUES (?,?,?,?,?,?,?)",
            (eid, branch, seq, kind, json.dumps(payload, ensure_ascii=False, sort_keys=True),
             _now(), undo_of))
        return eid, seq

    def add_event(self, branch, kind, payload, undo_of=None):
        eid, seq = self._add_event(branch, kind, payload, undo_of)
        self.conn.commit()
        return eid, seq

    def head(self, branch):
        row = self.conn.execute(
            "SELECT id, seq FROM events WHERE branch=? ORDER BY seq DESC LIMIT 1",
            (branch,)).fetchone()
        if row:
            return row["id"], int(row["seq"])
        return None, 0

    def events(self, branch, up_to_seq=None):
        q = "SELECT * FROM events WHERE branch=?"
        args = [branch]
        if up_to_seq is not None:
            q += " AND seq<=?"
            args.append(up_to_seq)
        q += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def get_event(self, eid):
        row = self.conn.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        return dict(row) if row else None

    # ----- derived values -----
    def put_derived(self, branch, dkey, deps, value, event_id, created_seq=None):
        seq = self._branch_seq(branch) if created_seq is None else created_seq
        self.conn.execute(
            "UPDATE derived SET invalidated_seq=? WHERE branch=? AND dkey=? AND invalidated_seq IS NULL",
            (seq, branch, dkey))
        version = int(self.conn.execute(
            "SELECT COUNT(*) AS c FROM derived WHERE branch=? AND dkey=?",
            (branch, dkey)).fetchone()["c"]) + 1
        did = self._insert_derived(branch, dkey, deps, value, event_id, seq, version)
        self.conn.commit()
        return did

    def _insert_derived(self, branch, dkey, deps, value, event_id, created_seq, version):
        did = f"der-{branch}-{_slug(dkey)}-v{version}"
        self.conn.execute(
            "INSERT INTO derived (id, branch, dkey, version, deps, value, event_id, created_seq,"
            " invalidated_seq) VALUES (?,?,?,?,?,?,?,?,NULL)",
            (did, branch, dkey, version, json.dumps(sorted(deps)),
             json.dumps(value, ensure_ascii=False), event_id, created_seq))
        return did

    def visible_derived(self, branch, at_seq=None):
        out = []
        for r in self.conn.execute(
                "SELECT * FROM derived WHERE branch=? ORDER BY created_seq, id", (branch,)):
            if at_seq is not None and r["created_seq"] > at_seq:
                continue
            if r["invalidated_seq"] is not None and (at_seq is None or r["invalidated_seq"] <= at_seq):
                continue
            out.append(dict(r))
        return out

    def derived_matching_deps(self, branch, dep_keys, at_seq=None):
        return [r for r in self.visible_derived(branch, at_seq)
                if set(dep_keys) & set(json.loads(r["deps"]))]

    def invalidate_by_deps(self, branch, dep_keys, seq):
        ids = [r["id"] for r in self.derived_matching_deps(branch, dep_keys)]
        self.conn.executemany("UPDATE derived SET invalidated_seq=? WHERE id=?",
                              [(seq, i) for i in ids])
        self.conn.commit()
        return ids

    # ----- rulings -----
    def effective_rulings(self, branch, up_to_seq=None):
        state = {}
        for ev in self.events(branch, up_to_seq):
            payload = json.loads(ev["payload"])
            if ev["kind"] == "fork":
                for subj, ruling in payload.get("inherited_rulings", {}).items():
                    state[subj] = ruling
            elif ev["kind"] == "ruling":
                state[payload["subject"]] = {
                    "event_id": ev["id"], "subject": payload["subject"],
                    "decision": payload["decision"], "rationale": payload.get("rationale", "")}
            elif ev["kind"] == "undo":
                undone = ev.get("undo_of")
                for subj in [s for s, r in state.items() if r.get("event_id") == undone]:
                    del state[subj]
        return state

    def view_at(self, branch, seq):
        return {"branch": branch, "at_seq": seq,
                "derived": self.visible_derived(branch, seq),
                "rulings": self.effective_rulings(branch, seq),
                "events": self.events(branch, seq)}

    # ----- snapshots -----
    def export_snapshot(self):
        return {"schema_version": self.schema_version(),
                "tables": {t: [dict(r) for r in self.conn.execute(f"SELECT * FROM {t} ORDER BY rowid")]
                           for t in TABLES}}

    def wipe(self):
        for t in TABLES:
            self.conn.execute(f"DELETE FROM {t}")
        self.conn.commit()

    def import_snapshot(self, snap):
        if int(snap.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"snapshot schema v{snap.get('schema_version')} != v{SCHEMA_VERSION}")
        self.wipe()
        for t in TABLES:
            for row in snap["tables"].get(t, []):
                cols = ",".join(row.keys())
                self.conn.execute(
                    f"INSERT INTO {t} ({cols}) VALUES ({','.join('?' for _ in row)})",
                    list(row.values()))
        self.conn.commit()
