"""SQLite 存储层: 模式迁移 + 连接.

表结构分三个迁移版本演进, 便于状态迁移测试:
  v1: counters / sources / events / branches
  v2: decisions / derived
  v3: judgments + sources.diagnostics
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 3

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE counters(
      name TEXT PRIMARY KEY,
      value INTEGER NOT NULL
    );
    CREATE TABLE sources(
      id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      ord INTEGER NOT NULL DEFAULT 0,
      imported_event TEXT,
      raw BLOB NOT NULL,
      sha256 TEXT NOT NULL,
      meta TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
    );
    CREATE TABLE events(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      id TEXT NOT NULL UNIQUE,
      branch_id TEXT,
      kind TEXT NOT NULL,
      payload TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
    );
    CREATE TABLE branches(
      id TEXT PRIMARY KEY,
      parent_id TEXT,
      created_event TEXT,
      version INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'open',
      created_at TEXT NOT NULL
    );
    """),
    (2, """
    CREATE TABLE decisions(
      id TEXT PRIMARY KEY,
      branch_id TEXT NOT NULL,
      target_kind TEXT NOT NULL,
      target_id TEXT NOT NULL,
      verdict TEXT NOT NULL,
      payload TEXT NOT NULL DEFAULT '{}',
      event_id TEXT NOT NULL,
      base_version INTEGER NOT NULL,
      undone_event TEXT,
      created_at TEXT NOT NULL
    );
    CREATE TABLE derived(
      id TEXT PRIMARY KEY,
      branch_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      deps TEXT NOT NULL DEFAULT '{}',
      payload TEXT NOT NULL DEFAULT '{}',
      status TEXT NOT NULL DEFAULT 'valid',
      created_event TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    """),
    (3, """
    CREATE TABLE judgments(
      id TEXT PRIMARY KEY,
      branch_id TEXT,
      rule TEXT NOT NULL,
      subject_id TEXT,
      payload TEXT NOT NULL DEFAULT '{}',
      event_id TEXT,
      created_at TEXT NOT NULL
    );
    ALTER TABLE sources ADD COLUMN diagnostics TEXT NOT NULL DEFAULT '[]';
    """),
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations("
        "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    have = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    for version, sql in MIGRATIONS:
        if version not in have:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, utcnow()),
            )
    conn.commit()


def connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn
