"""事件日志：分支、裁定撤销、版本冲突、时间旅行、失效范围。"""
import pytest

from waveforge import store

from conftest import CASE


def boot(conn):
    store.create_branch(conn, "br:main", "main")
    versions = {"rule_version": "align-rules:v1", "params_version": "params:v1",
                "interval_version": "intervals:v1"}
    store.append_event(conn, "br:main", "import-baseline", {"case_id": CASE}, 0, versions)
    store.append_event(conn, "br:main", "set-intervals",
                       {"interval_version": "intervals:v1"}, 1)


def test_version_conflict_carries_both_diffs(loaded):
    boot(loaded)
    store.append_event(loaded, "br:main", "ruling",
                       {"target": "cand:seg:ch1/0004/0", "decision": "accept"}, 2)
    with pytest.raises(store.VersionConflict) as exc:
        store.append_event(loaded, "br:main", "ruling",
                           {"target": "cand:seg:ch1/0004/1", "decision": "accept"}, 2)
    e = exc.value
    assert e.base_seq == 2 and e.head_seq == 3
    assert e.their_events[0]["payload"]["target"] == "cand:seg:ch1/0004/0"
    assert e.your_event["payload"]["target"] == "cand:seg:ch1/0004/1"


def test_undo_restores_previous_state(loaded):
    boot(loaded)
    evt = store.append_event(loaded, "br:main", "ruling",
                             {"target": "cand:seg:ch1/0004/1", "decision": "accept"}, 2)
    state = store.fold_state(loaded, "br:main")
    assert state["rulings"]["cand:seg:ch1/0004/1"] == "accept"
    store.undo_event(loaded, "br:main", evt["id"], 3)
    state = store.fold_state(loaded, "br:main")
    assert "cand:seg:ch1/0004/1" not in state["rulings"]  # 恢复前一分支状态
    events = store.list_events(loaded, "br:main")
    assert any(e["kind"] == "undo" for e in events)
    assert next(e for e in events if e["id"] == evt["id"])["undone"] == 1


def test_time_travel_restores_versions(loaded):
    boot(loaded)
    store.append_event(loaded, "br:main", "set-params",
                       {"params_version": "params:v2"}, 2)
    now = store.fold_state(loaded, "br:main")
    assert now["params_version"] == "params:v2"
    then = store.fold_state(loaded, "br:main", upto_seq=2)
    assert then["params_version"] == "params:v1"  # 回到旧事件恢复当时依赖版本


def test_branch_keeps_independent_rulings(loaded):
    boot(loaded)
    store.create_branch(loaded, "br:b", "B", parent_id="br:main", parent_seq=2)
    store.append_event(loaded, "br:main", "ruling",
                       {"target": "cand:x/0", "decision": "accept"}, 2)
    store.append_event(loaded, "br:b", "ruling",
                       {"target": "cand:x/1", "decision": "accept"}, 0)
    assert store.fold_state(loaded, "br:main")["rulings"] == {"cand:x/0": "accept"}
    assert store.fold_state(loaded, "br:b")["rulings"] == {"cand:x/1": "accept"}


def test_invalidation_scope_is_explicit(scenariod):
    # 参数修订：对齐与测量都失效
    scope = store.invalidation_scope(scenariod, "br:main", "params:v1")
    kinds = {s["kind"] for s in scope}
    assert kinds == {"align", "meas"}
    # 源数据局部改动：只有 ch2/0005 失效
    scope = store.invalidation_scope(scenariod, "br:main",
                                     "src:wave-overlay-demo/segments/ch2_0005")
    assert scope and all(s["target"].startswith("seg:ch2/0005") for s in scope)
    # 旧派生保留为 stale，不被全局悄悄替换
    stale = [d for d in store.stale_derivations(scenariod, "br:main") if d["stale"]]
    assert stale
