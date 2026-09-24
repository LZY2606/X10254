"""State migration tests: old on-disk schemas upgrade without losing data."""
import json
import sqlite3

import pytest

from appcore.casegen import build_case
from appcore.ingest import ingest_case
from appcore.projection import project
from appcore.repository import SCHEMA_VERSION, Repository


def test_fresh_db_has_current_schema_version(tmp_path):
    repo = Repository(str(tmp_path / "x.sqlite3"))
    assert repo.schema_version() == SCHEMA_VERSION
    repo.close()


def test_migrates_from_v1_legacy_layout(tmp_path):
    """v1 had no derived_cache and a plain meta layout; data must survive."""
    path = tmp_path / "old.sqlite3"
    case = build_case()
    # create current db with an ingest, then rewrite it down to the v1 shape
    repo = Repository(str(path))
    ingest_case(repo, case)
    expected_events = repo.all_events()
    expected_materials = repo.list_materials()
    repo.close()

    conn = sqlite3.connect(str(path))
    conn.execute("DROP TABLE IF EXISTS derived_cache")
    conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    migrated = Repository(str(path))
    assert migrated.schema_version() == SCHEMA_VERSION
    rows = migrated.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='derived_cache'").fetchall()
    assert rows
    assert [e["event_id"] for e in migrated.all_events()] == \
           [e["event_id"] for e in expected_events]
    assert {m["material_id"] for m in migrated.list_materials()} == \
           {m["material_id"] for m in expected_materials}
    state = project(migrated.branch_events("main"), case)
    assert len(state["segments"]) == 8
    migrated.close()


def test_migrates_from_v2_without_branch_index(tmp_path):
    path = tmp_path / "v2.sqlite3"
    repo = Repository(str(path))
    ingest_case(repo, build_case())
    repo.close()
    conn = sqlite3.connect(str(path))
    conn.execute("DROP INDEX IF EXISTS idx_cache_event")
    conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    migrated = Repository(str(path))
    assert migrated.schema_version() == SCHEMA_VERSION
    assert migrated.branch_events("main")
    migrated.close()


def test_corrupt_bundle_checksum_is_rejected_and_db_untouched(tmp_path):
    case = build_case()
    repo = Repository(":memory:")
    ingest_case(repo, case)
    bundle = repo.export_bundle()
    events_before = repo.all_events()
    # corrupt a blob
    import base64
    raw = bytearray(base64.b64decode(bundle["blobs"][0]["content_b64"]))
    raw[0] ^= 0xFF
    bundle["blobs"][0]["content_b64"] = base64.b64encode(bytes(raw)).decode()

    empty = Repository(":memory:")
    with pytest.raises(ValueError):
        empty.import_bundle(bundle)
    assert empty.all_events() == []
    assert empty.list_materials() == []
