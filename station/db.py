"""SQLite 存储：material/result 内容寻址、branch/event 追加式事件链。

导入失败整批回滚（SAVEPOINT/事务），不留下半份记录。
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable

SCHEMA_VERSION = 2

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        case_id TEXT,
        created_at TEXT NOT NULL
    )""",
    # 内容寻址的来源材料（原始表示 + 单位/坐标基准），重复导入为同一行
    """
    CREATE TABLE IF NOT EXISTS materials (
        material_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        segment_no INTEGER,
        channel TEXT,
        kind TEXT NOT NULL,
        version_label TEXT,
        payload TEXT NOT NULL,
        raw_blob_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    # 内容寻址的派生结果（规则版本 + 依赖 + 主体）
    """
    CREATE TABLE IF NOT EXISTS derivations (
        result_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        subject TEXT NOT NULL,
        rule_version TEXT NOT NULL,
        body TEXT NOT NULL,
        deps TEXT NOT NULL
    )""",
    # 派生依赖边（material|result -> result），用于失效范围闭包
    """
    CREATE TABLE IF NOT EXISTS derivation_deps (
        from_id TEXT NOT NULL,
        to_result_id TEXT NOT NULL,
        layer TEXT NOT NULL,
        PRIMARY KEY (from_id, to_result_id)
    )""",
    """
    CREATE TABLE IF NOT EXISTS branches (
        branch_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        name TEXT NOT NULL,
        parent_branch_id TEXT,
        root_event_id TEXT,
        created_at TEXT NOT NULL
    )""",
    # 追加式事件链；seq 分支内递增，prev_event_id 形成哈希链
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        branch_id TEXT NOT NULL,
        branch_seq INTEGER NOT NULL,
        prev_event_id TEXT,
        type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (branch_id, branch_seq)
    )""",
    # 机器可读取的判定事件（J1 物理时间 / J2 缺样饱和 / J3 多峰候选 ...）
    """
    CREATE TABLE IF NOT EXISTS judgments (
        judgment_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL,
        branch_id TEXT NOT NULL,
        code TEXT NOT NULL,
        subject TEXT,
        detail TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    # 反例档案试运行：采纳/未采纳与期望/实际失效范围
    """
    CREATE TABLE IF NOT EXISTS counter_trials (
        trial_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        branch_id TEXT,
        name TEXT NOT NULL,
        mutated_source TEXT NOT NULL,
        adopted INTEGER NOT NULL,
        expected_scope TEXT NOT NULL,
        actual_scope TEXT NOT NULL,
        judgments TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_mat_project ON materials(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_dep_from ON derivation_deps(from_id)",
    "CREATE INDEX IF NOT EXISTS idx_dep_to ON derivation_deps(to_result_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_branch ON events(branch_id, branch_seq)",
    "CREATE INDEX IF NOT EXISTS idx_judgment_branch ON judgments(branch_id)",
]

_MIGRATIONS = {
    2: [
        "ALTER TABLE projects ADD COLUMN case_title TEXT",
    ],
}


def connect(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or default_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def default_db_path() -> str:
    return os.environ.get("STATION_DB", os.path.join(os.getcwd(), "station.db"))


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        current = int(row["value"]) if row else 0
        for stmt in _SCHEMA:
            conn.execute(stmt)
        for ver, stmts in sorted(_MIGRATIONS.items()):
            if current < ver:
                for stmt in stmts:
                    conn.execute(stmt)
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


@contextmanager
def savepoint(conn: sqlite3.Connection, name: str):
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield conn
        conn.execute(f"RELEASE SAVEPOINT {name}")
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
        raise


def jdumps(obj: Any) -> str:
    from station.ids import canonical_json

    return canonical_json(obj)


def jloads(text: str) -> Any:
    return json.loads(text)
