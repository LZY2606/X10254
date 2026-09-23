"""SQLite 存储层：schema、迁移。

schema_version 历史：
  1 -> 初始版本（无 diagnostics 表，events 无 undone 列）
  2 -> 当前版本（增加 diagnostics 表与 events.undone）
"""
import json
import os
import sqlite3
import time

SCHEMA_VERSION = 2

DDL_V2 = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  ord INTEGER NOT NULL,
  raw TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  imported_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS diagnostics (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES sources(id),
  path TEXT NOT NULL,
  severity TEXT NOT NULL,
  message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS branches (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  parent_id TEXT,
  parent_seq INTEGER,
  created_gseq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  gseq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  branch_id TEXT NOT NULL REFERENCES branches(id),
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  rule_version TEXT,
  params_version TEXT,
  interval_version TEXT,
  undone INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  UNIQUE(branch_id, seq)
);
CREATE TABLE IF NOT EXISTS derivations (
  id TEXT PRIMARY KEY,
  branch_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  target TEXT NOT NULL,
  versions TEXT NOT NULL,
  inputs TEXT NOT NULL,
  output TEXT NOT NULL,
  stale INTEGER NOT NULL DEFAULT 0,
  created_gseq INTEGER NOT NULL
);
"""

# 历史 v1 schema（仅用于迁移测试与旧库升级）
DDL_V1 = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY, case_id TEXT NOT NULL, kind TEXT NOT NULL,
  name TEXT NOT NULL, ord INTEGER NOT NULL, raw TEXT NOT NULL,
  sha256 TEXT NOT NULL, imported_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS branches (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, parent_id TEXT,
  parent_seq INTEGER, created_gseq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  gseq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
  branch_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
  payload TEXT NOT NULL, rule_version TEXT, params_version TEXT,
  interval_version TEXT, created_at REAL NOT NULL,
  UNIQUE(branch_id, seq)
);
CREATE TABLE IF NOT EXISTS derivations (
  id TEXT PRIMARY KEY, branch_id TEXT NOT NULL, kind TEXT NOT NULL,
  target TEXT NOT NULL, versions TEXT NOT NULL, inputs TEXT NOT NULL,
  output TEXT NOT NULL, stale INTEGER NOT NULL DEFAULT 0,
  created_gseq INTEGER NOT NULL
);
"""

MIGRATIONS = {
    # 1 -> 2: 增加 diagnostics 表与 events.undone 列
    1: [
        """CREATE TABLE IF NOT EXISTS diagnostics (
             id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
             path TEXT NOT NULL, severity TEXT NOT NULL, message TEXT NOT NULL)""",
        "ALTER TABLE events ADD COLUMN undone INTEGER NOT NULL DEFAULT 0",
    ],
}


def connect(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn):
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["value"]) if row else 0


def init_db(conn):
    conn.executescript(DDL_V2)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def migrate(conn):
    """把旧版本库迁移到当前版本；返回应用的迁移列表。"""
    applied = []
    version = schema_version(conn)
    if version == 0:
        init_db(conn)
        return applied
    while version < SCHEMA_VERSION:
        for stmt in MIGRATIONS[version]:
            conn.execute(stmt)
        version += 1
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(version),),
        )
        applied.append(version)
    conn.commit()
    return applied


def open_db(path):
    """打开（必要时创建/迁移）数据库。"""
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    conn = connect(path)
    if fresh:
        init_db(conn)
    else:
        migrate(conn)
    return conn


def now():
    return time.time()


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def loads(text):
    return json.loads(text)
