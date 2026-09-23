"""SQLite 存储层 + 模式迁移。

设计要点:
  - sources 存原始字节(原始表示), 永不改写; 重新解释 = 新分支。
  - events 是唯一的变更入口(append-only), seq 即全局顺序与稳定身份。
  - 派生结论不落库, 由 core 重放事件现算; 因此导出/导入只需搬运
    sources/branches/events 三张表即可完整保持身份、顺序与来源关系。
"""
from __future__ import annotations

import json
import sqlite3
import time

SCHEMA_VERSION = 2

MIGRATIONS = {
    1: """
    CREATE TABLE sources (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE branches (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        parent_id TEXT,
        base_seq INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    CREATE TABLE events (
        seq INTEGER PRIMARY KEY,
        id TEXT NOT NULL UNIQUE,
        branch_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        payload TEXT NOT NULL,
        actor TEXT NOT NULL DEFAULT 'reviewer',
        created_at TEXT NOT NULL
    );
    """,
    # v2: 事件增加会话列(区分两个复核窗口), 新增键值表存案例包指针等
    2: """
    ALTER TABLE events ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default';
    CREATE TABLE kv (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
}


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def current_version(conn) -> int:
    try:
        row = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
        return row["v"] or 0
    except sqlite3.OperationalError:
        return 0


def migrate(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    v = current_version(conn)
    while v < SCHEMA_VERSION:
        v += 1
        with conn:  # 单步迁移失败整体回滚, 不留半份模式
            conn.executescript(MIGRATIONS[v])
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (v, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )


def insert_source(conn, sid, kind, name, content: str, content_hash):
    with conn:
        conn.execute(
            "INSERT INTO sources(id, kind, name, content, content_hash, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (sid, kind, name, content, content_hash,
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )


def insert_branch(conn, bid, name, parent_id, base_seq):
    with conn:
        conn.execute(
            "INSERT INTO branches(id, name, parent_id, base_seq, created_at)"
            " VALUES (?,?,?,?,?)",
            (bid, name, parent_id, base_seq,
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )


def append_event(conn, branch_id, kind, payload: dict, actor="reviewer",
                 session_id="default", seq=None, event_id=None, created_at=None):
    """追加事件; 导入归档时用显式 seq/id 保持原身份与顺序。"""
    created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with conn:
        if seq is None:
            row = conn.execute(
                "INSERT INTO events(branch_id, kind, payload, actor, session_id, created_at)"
                " VALUES (?,?,?,?,?,?) RETURNING seq",
                (branch_id, kind, json.dumps(payload, ensure_ascii=False, sort_keys=True),
                 actor, session_id, created_at),
            ).fetchone()
            seq = row["seq"]
            event_id = event_id or f"evt-{seq:06d}"
            conn.execute("UPDATE events SET id=? WHERE seq=?", (event_id, seq))
        else:
            if event_id is None:
                raise ValueError("explicit seq requires explicit event id")
            conn.execute(
                "INSERT INTO events(seq, id, branch_id, kind, payload, actor, session_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (seq, event_id, branch_id, kind,
                 json.dumps(payload, ensure_ascii=False, sort_keys=True),
                 actor, session_id, created_at),
            )
    return seq, event_id


def branch_head(conn, branch_id) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) AS h FROM events WHERE branch_id=?", (branch_id,)
    ).fetchone()
    return row["h"]


def dump_tables(conn) -> dict:
    def rows(table):
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
    return {
        "schema_version": current_version(conn),
        "sources": rows("sources"),
        "branches": rows("branches"),
        "events": sorted(rows("events"), key=lambda e: e["seq"]),
        "kv": rows("kv"),
    }


def wipe(conn) -> None:
    with conn:
        for t in ("events", "branches", "sources", "kv"):
            conn.execute(f"DELETE FROM {t}")


def load_dump(conn, dump: dict) -> None:
    """导入归档: 全部成功才提交, 任一失败整体回滚。"""
    with conn:
        for s in dump["sources"]:
            conn.execute(
                "INSERT INTO sources(id, kind, name, content, content_hash, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (s["id"], s["kind"], s["name"], s["content"], s["content_hash"], s["created_at"]),
            )
        for b in dump["branches"]:
            conn.execute(
                "INSERT INTO branches(id, name, parent_id, base_seq, created_at)"
                " VALUES (?,?,?,?,?)",
                (b["id"], b["name"], b["parent_id"], b["base_seq"], b["created_at"]),
            )
        for e in sorted(dump["events"], key=lambda x: x["seq"]):
            conn.execute(
                "INSERT INTO events(seq, id, branch_id, kind, payload, actor, session_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (e["seq"], e["id"], e["branch_id"], e["kind"], e["payload"],
                 e.get("actor", "reviewer"), e.get("session_id", "default"), e["created_at"]),
            )
        for k in dump.get("kv", []):
            conn.execute("INSERT INTO kv(key, value) VALUES (?,?)", (k["key"], k["value"]))
