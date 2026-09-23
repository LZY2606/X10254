"""SQLite 事件溯源存储。

原则：
- 来源（sources，含原始字节/原始表示）与解释（events/branches）分开保存；
- 事件只追加，裁定可撤销（undo/restore 也是事件），不做不可逆清洗；
- 导入失败只回滚本次变更，不留下半份记录；
- 保存携带 expected_head，版本冲突时后提交者看到双方差异；
- schema 迁移显式版本化（schema_version 表）。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3

SCHEMA_VERSION = 2

SCHEMA_V1 = """
CREATE TABLE sources(
    source_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    PRIMARY KEY(source_id, version)
);
CREATE TABLE branches(
    branch_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT,
    base_seq INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE events(
    event_id TEXT PRIMARY KEY,
    branch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

SCHEMA_V2_EXTRA = """
CREATE TABLE derived(
    branch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    key TEXT NOT NULL,
    deps TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(branch_id, key)
);
CREATE TABLE schema_version(version INTEGER NOT NULL);
"""


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _hash(payload_text):
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def connect(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return set(r["name"] for r in rows)


def init_db(conn):
    """创建当前 schema；若是 v1 库则迁移到 v2（状态迁移）。"""
    tables = _tables(conn)
    if "schema_version" in tables:
        return
    if "sources" in tables:  # v1 库：迁移
        with conn:
            conn.execute("ALTER TABLE sources ADD COLUMN content_hash TEXT")
            conn.executescript(SCHEMA_V2_EXTRA)
            rows = conn.execute("SELECT source_id, version, payload FROM sources").fetchall()
            for r in rows:
                conn.execute(
                    "UPDATE sources SET content_hash=? WHERE source_id=? AND version=?",
                    (_hash(r["payload"]), r["source_id"], r["version"]))
            conn.execute("INSERT INTO schema_version(version) VALUES(?)", (SCHEMA_VERSION,))
        return
    with conn:
        conn.executescript(SCHEMA_V1)
        conn.execute("ALTER TABLE sources ADD COLUMN content_hash TEXT")
        conn.executescript(SCHEMA_V2_EXTRA)
        conn.execute("INSERT INTO schema_version(version) VALUES(?)", (SCHEMA_VERSION,))


def schema_version(conn):
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return row["version"] if row else None


class Conflict(Exception):
    """版本冲突：携带当前头、双方差异，供后提交者审阅。"""

    def __init__(self, current_head, your_base, events_since, your_payload):
        self.current_head = current_head
        self.your_base = your_base
        self.events_since = events_since
        self.your_payload = your_payload
        super().__init__("version conflict at head %s" % current_head)

    def to_dict(self):
        return {
            "error": "version-conflict",
            "current_head": self.current_head,
            "your_base": self.your_base,
            "events_since": self.events_since,
            "your_payload": self.your_payload,
        }


# ------------------------------------------------------------- sources

_REQUIRED = {
    "segment": ["segment_id", "channel", "samples", "timebase",
                "vertical_gain", "probe_attenuation", "trigger_time"],
    "channel": ["channel_id", "deskew"],
    "interval-set": ["versions", "active"],
    "case-meta": ["case_id", "title"],
}


def _validate(kind, payload):
    for field in _REQUIRED.get(kind, []):
        if field not in payload:
            raise ValueError("source kind %r missing field %r" % (kind, field))
    if kind == "segment":
        if not isinstance(payload["samples"].get("values"), list):
            raise ValueError("segment samples.values must be a list")
        if payload["timebase"].get("kind") not in ("uniform", "explicit"):
            raise ValueError("unknown timebase kind")


def import_sources(conn, items):
    """事务性导入；任一来源非法则整体回滚，不留半份记录。"""
    with conn:
        for source_id, version, kind, payload in items:
            _validate(kind, payload)
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            conn.execute(
                "INSERT INTO sources(source_id, version, kind, payload, imported_at, content_hash)"
                " VALUES(?,?,?,?,?,?)",
                (source_id, version, kind, text, _now(), _hash(text)))


def put_source_version(conn, source_id, version, kind, payload):
    _validate(kind, payload)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    conn.execute(
        "INSERT INTO sources(source_id, version, kind, payload, imported_at, content_hash)"
        " VALUES(?,?,?,?,?,?)",
        (source_id, version, kind, text, _now(), _hash(text)))


def has_sources(conn):
    row = conn.execute("SELECT COUNT(*) AS c FROM sources").fetchone()
    return row["c"] > 0


def get_sources(conn):
    out = {}
    rows = conn.execute(
        "SELECT source_id, version, kind, payload, content_hash FROM sources"
        " ORDER BY source_id, version").fetchall()
    for r in rows:
        entry = out.setdefault(r["source_id"], {"kind": r["kind"], "versions": {},
                                                "content_hashes": {}})
        entry["versions"][r["version"]] = json.loads(r["payload"])
        entry["content_hashes"][r["version"]] = r["content_hash"]
    return out


# ------------------------------------------------------------- branches / events

def create_branch(conn, branch_id, name, parent_id=None):
    base_seq = head_seq(conn, parent_id) if parent_id else 0
    with conn:
        conn.execute(
            "INSERT INTO branches(branch_id, name, parent_id, base_seq, created_at)"
            " VALUES(?,?,?,?,?)",
            (branch_id, name, parent_id, base_seq, _now()))
    return {"branch_id": branch_id, "name": name, "parent_id": parent_id,
            "base_seq": base_seq}


def list_branches(conn):
    rows = conn.execute(
        "SELECT branch_id, name, parent_id, base_seq, created_at FROM branches"
        " ORDER BY created_at, branch_id").fetchall()
    return [dict(r) for r in rows]


def _get_branch(conn, branch_id):
    row = conn.execute(
        "SELECT branch_id, name, parent_id, base_seq FROM branches WHERE branch_id=?",
        (branch_id,)).fetchone()
    if not row:
        raise KeyError("unknown branch %r" % branch_id)
    return dict(row)


def gather_events(conn, branch_id):
    """沿父链收集该分支可见的事件（fork 继承父分支 base_seq 之前的事件）。"""
    br = _get_branch(conn, branch_id)
    out = []
    if br["parent_id"]:
        out.extend(e for e in gather_events(conn, br["parent_id"])
                   if e["seq"] <= br["base_seq"])
    rows = conn.execute(
        "SELECT event_id, branch_id, seq, type, payload, created_at FROM events"
        " WHERE branch_id=? ORDER BY seq", (branch_id,)).fetchall()
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        out.append(d)
    out.sort(key=lambda e: e["seq"])
    return out


def head_seq(conn, branch_id):
    events = gather_events(conn, branch_id)
    return events[-1]["seq"] if events else 0


def add_event(conn, branch_id, type_, payload, expected_head=None):
    head = head_seq(conn, branch_id)
    if expected_head is not None and expected_head != head:
        since = [e for e in gather_events(conn, branch_id) if e["seq"] > expected_head]
        raise Conflict(head, expected_head, since, payload)
    seq = head + 1
    event_id = "ev-%s-%04d" % (branch_id, seq)
    with conn:
        conn.execute(
            "INSERT INTO events(event_id, branch_id, seq, type, payload, created_at)"
            " VALUES(?,?,?,?,?,?)",
            (event_id, branch_id, seq, type_,
             json.dumps(payload, ensure_ascii=False, sort_keys=True), _now()))
    return {"event_id": event_id, "branch_id": branch_id, "seq": seq,
            "type": type_, "payload": payload}


# ------------------------------------------------------------- derived snapshots

def replace_derived(conn, branch_id, seq, derived):
    """显式保存派生层（多候选、抖动、测量）及其依赖，供失效范围计算。"""
    with conn:
        conn.execute("DELETE FROM derived WHERE branch_id=?", (branch_id,))
        for key, entry in derived.items():
            conn.execute(
                "INSERT INTO derived(branch_id, seq, key, deps, payload) VALUES(?,?,?,?,?)",
                (branch_id, seq, key,
                 json.dumps(entry["deps"], ensure_ascii=False, sort_keys=True),
                 json.dumps(entry["payload"], ensure_ascii=False, sort_keys=True)))


def get_derived(conn, branch_id):
    rows = conn.execute(
        "SELECT key, deps, payload FROM derived WHERE branch_id=? ORDER BY key",
        (branch_id,)).fetchall()
    return {r["key"]: {"deps": json.loads(r["deps"]), "payload": json.loads(r["payload"])}
            for r in rows}


# ------------------------------------------------------------- export / import

def export_db(conn):
    """导出全部表；身份、顺序与来源关系由原样主键保留。"""
    def dump(table, order):
        rows = conn.execute("SELECT * FROM %s ORDER BY %s" % (table, order)).fetchall()
        return [dict(r) for r in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": dump("sources", "source_id, version"),
        "branches": dump("branches", "created_at, branch_id"),
        "events": dump("events", "seq, event_id"),
        "derived": dump("derived", "branch_id, key"),
    }


def import_db(conn, data):
    """导入到空库；非空库拒绝，导入失败整体回滚。"""
    if has_sources(conn):
        raise Conflict(0, 0, [], {"error": "database not empty"})
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version %r" % (data.get("schema_version"),))
    with conn:
        for r in data["sources"]:
            conn.execute(
                "INSERT INTO sources VALUES(?,?,?,?,?,?)",
                (r["source_id"], r["version"], r["kind"], r["payload"],
                 r["imported_at"], r["content_hash"]))
        for r in data["branches"]:
            conn.execute(
                "INSERT INTO branches VALUES(?,?,?,?,?)",
                (r["branch_id"], r["name"], r["parent_id"], r["base_seq"], r["created_at"]))
        for r in data["events"]:
            conn.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?)",
                (r["event_id"], r["branch_id"], r["seq"], r["type"],
                 r["payload"], r["created_at"]))
        for r in data["derived"]:
            conn.execute(
                "INSERT INTO derived VALUES(?,?,?,?,?)",
                (r["branch_id"], r["seq"], r["key"], r["deps"], r["payload"]))
