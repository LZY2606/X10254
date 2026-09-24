"""SQLite schema definitions with an explicit v1 -> v2 migration path."""

SCHEMA_VERSION = 2

_DDL_V1 = """
CREATE TABLE blobs (
    id          TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    media_type  TEXT NOT NULL,
    length      INTEGER NOT NULL,
    content     BLOB NOT NULL
);

CREATE TABLE import_batches (
    id              TEXT PRIMARY KEY,
    source_file     TEXT NOT NULL,
    media_type      TEXT NOT NULL,
    blob_id         TEXT NOT NULL REFERENCES blobs(id),
    created_event_id TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE materials (
    id          TEXT PRIMARY KEY,
    batch_id    TEXT NOT NULL REFERENCES import_batches(id),
    grp         TEXT NOT NULL,
    key         TEXT NOT NULL,
    version     INTEGER NOT NULL,
    ord         INTEGER NOT NULL,
    data_json   TEXT NOT NULL,
    UNIQUE(grp, key, version)
);

CREATE TABLE branches (
    name            TEXT PRIMARY KEY,
    fork_event_id   TEXT,
    head_event_id   TEXT NOT NULL,
    created_event_id TEXT NOT NULL,
    label           TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE events (
    id               TEXT PRIMARY KEY,
    ord              INTEGER NOT NULL UNIQUE,
    branch           TEXT NOT NULL,
    type             TEXT NOT NULL,
    parent_event_id  TEXT,
    created_at       TEXT NOT NULL,
    payload_json     TEXT NOT NULL
);

CREATE TABLE kv (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""

_DDL_V2 = """
CREATE TABLE annotations (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL REFERENCES events(id),
    author      TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def initialize(conn, version: int = SCHEMA_VERSION) -> None:
    """Create schema at the requested version (for migration tests)."""
    conn.execute("PRAGMA foreign_keys = ON")
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current == 0:
        conn.executescript(_DDL_V1)
        current = 1
        conn.execute("PRAGMA user_version = 1")
    if current == 1 and version >= 2:
        conn.executescript(_DDL_V2)
        conn.execute("PRAGMA user_version = 2")
        current = 2
    conn.commit()


def current_version(conn) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]
