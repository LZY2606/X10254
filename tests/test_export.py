"""导出 -> 删除本地库 -> 导入：身份、顺序、来源关系保持。"""
import os

from waveforge import db as dbmod, exporter, store

from conftest import CASE, PKG
from waveforge import ingest, scenario


def test_export_delete_import_roundtrip(tmp_path):
    path = str(tmp_path / "a.db")
    conn = dbmod.open_db(path)
    ingest.load_case_package(conn, PKG)
    for _ in range(7):
        scenario.apply_next_step(conn, PKG, CASE)
    conn.commit()
    archive = exporter.export_archive(conn)
    view_before = store.compute_branch_view(conn, CASE, "br:main")
    conn.close()
    os.remove(path)  # 删除本地数据库

    conn2 = dbmod.open_db(path)
    counts = exporter.import_archive(conn2, archive)
    assert counts["events"] > 0 and counts["sources"] == 12
    archive2 = exporter.export_archive(conn2)
    assert archive == archive2  # 完全往返
    # 身份与顺序
    ev = [e["id"] for e in store.list_events(conn2, "br:main")]
    assert ev == sorted(ev, key=lambda x: int(x.rsplit("/", 1)[1]))
    srcs = ingest.get_sources(conn2, case_id=CASE)
    assert [s["ord"] for s in srcs] == sorted(s["ord"] for s in srcs)
    # 来源关系（source-patch 覆盖链）仍可追溯
    view_after = store.compute_branch_view(conn2, CASE, "br:main")
    assert view_after["versions"] == view_before["versions"]
    assert view_after["jitter"] == view_before["jitter"]
    evt = [e for e in store.list_events(conn2, "br:main")
           if e["kind"] == "source-patch"][0]
    import json
    p = json.loads(evt["payload"])
    assert p["source_id"].endswith("ch2_0005") and p["new_source_id"].endswith("rev2")


def test_reimport_over_populated_db(scenariod):
    """UI 的“删库重导”等价于在非空库上整体替换，必须成功且结果一致。"""
    archive = exporter.export_archive(scenariod)
    counts = exporter.import_archive(scenariod, archive)
    assert counts["events"] > 0
    assert exporter.export_archive(scenariod) == archive
