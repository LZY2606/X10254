"""存储层测试：导入回滚、冲突、undo/restore 重放、导出导入身份保持。"""
import json

import pytest

import casepack
import state as state_mod
import store


def _db(tmp_path, name="t.db"):
    conn = store.connect(str(tmp_path / name))
    store.init_db(conn)
    return conn


def _load_case(conn):
    data = casepack.load_baseline()
    store.import_sources(conn, casepack.baseline_sources(data))
    store.create_branch(conn, "br-main", "main")
    store.add_event(conn, "br-main", "import", {"case_id": data["case_id"]})
    state_mod.refresh_derived(conn, "br-main")


def test_import_failure_rolls_back_whole_batch(tmp_path):
    conn = _db(tmp_path)
    good = ("case-meta", 1, "case-meta", {"case_id": "x", "title": "t"})
    bad = ("seg-bad", 1, "segment", {"segment_id": "seg-bad"})  # 缺字段
    with pytest.raises(ValueError):
        store.import_sources(conn, [good, bad])
    assert not store.has_sources(conn)  # 不留半份记录


def test_conflict_carries_both_sides(tmp_path):
    conn = _db(tmp_path)
    _load_case(conn)
    store.add_event(conn, "br-main", "ruling", {"n": 1}, expected_head=1)
    with pytest.raises(store.Conflict) as exc:
        store.add_event(conn, "br-main", "ruling", {"n": 2}, expected_head=1)
    c = exc.value
    assert c.current_head == 2 and c.your_base == 1
    assert [e["payload"]["n"] for e in c.events_since] == [1]
    assert c.your_payload == {"n": 2}


def test_undo_and_restore_replay(tmp_path):
    conn = _db(tmp_path)
    _load_case(conn)
    e2 = store.add_event(conn, "br-main", "ruling",
                         {"segment_id": "seg-01", "candidate_id": "cand-seg-01-2",
                          "decision": "accept"})
    st = state_mod.compute_state(conn, "br-main")
    assert st["chosen"]["seg-01"] == "cand-seg-01-2"
    store.add_event(conn, "br-main", "undo", {"target_seq": e2["seq"]})
    st = state_mod.compute_state(conn, "br-main")
    assert st["chosen"]["seg-01"] == "cand-seg-01-1"  # 恢复前一分支结论
    # restore 到导入事件：任何后续裁定都不可见
    st = state_mod.compute_state(conn, "br-main", upto_seq=1)
    assert st["chosen"]["seg-01"] == "cand-seg-01-1"
    # 历史事件仍然可读
    assert any(e["type"] == "ruling" for e in st["events"])


def test_export_import_preserves_identity(tmp_path):
    conn1 = _db(tmp_path, "a.db")
    _load_case(conn1)
    store.add_event(conn1, "br-main", "ruling",
                    {"segment_id": "seg-01", "candidate_id": "cand-seg-01-2",
                     "decision": "accept"})
    state_mod.refresh_derived(conn1, "br-main")
    dump = store.export_db(conn1)

    conn2 = _db(tmp_path, "b.db")
    store.import_db(conn2, dump)
    dump2 = store.export_db(conn2)
    assert dump == dump2  # 身份、顺序、来源关系完全保持
    st1 = state_mod.compute_state(conn1, "br-main")
    st2 = state_mod.compute_state(conn2, "br-main")
    assert json.dumps(st1, sort_keys=True) == json.dumps(st2, sort_keys=True)


def test_import_into_nonempty_db_rejected(tmp_path):
    conn = _db(tmp_path)
    _load_case(conn)
    with pytest.raises(store.Conflict):
        store.import_db(conn, store.export_db(conn))
