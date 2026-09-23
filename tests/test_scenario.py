"""案例剧本与反例档案：结论、来源链、失效范围三重断言。"""
import json
import os

from waveforge import scenario, store

from conftest import CASE, PKG


def test_full_scenario_replay(scenariod):
    heads = {b: store.head_seq(scenariod, b)
             for b in ("br:main", "br:review-b", "br:edge-rules")}
    assert all(h > 0 for h in heads.values())
    view = store.compute_branch_view(scenariod, CASE, "br:main")
    assert view["versions"]["params_version"] == "params:v2"
    assert view["versions"]["interval_version"] == "intervals:v2"
    # 七个边界情形都在最终状态留下可读痕迹
    rules = {e["rule"] for e in view["rule_events"]}
    assert {"nonuniform-no-index-sub", "gap-no-interp", "multipeak-candidates",
            "trigger-at-boundary"} <= rules
    excluded = {e["segment"] for e in view["excluded"]}
    assert {"seg:ch1/0007", "seg:ch1/0008"} <= excluded


def test_conflicting_decisions_both_kept(scenariod):
    main = store.compute_branch_view(scenariod, CASE, "br:main")
    alt = store.compute_branch_view(scenariod, CASE, "br:review-b")
    sel_main = next(s for s in main["segments"] if s["id"] == "seg:ch1/0004")["selected"]
    sel_alt = next(s for s in alt["segments"] if s["id"] == "seg:ch1/0004")["selected"]
    assert sel_main["rank"] == 0 and sel_alt["rank"] == 1  # 采纳与未采纳都保留
    assert abs(sel_main["tau"] - sel_alt["tau"]) > 1e-9
    # 冲突事件可反查：主线上保留着被接受的裁定，B 分支保留着另一个决定
    rulings_main = store.fold_state(scenariod, "br:main")["rulings"]
    rulings_alt = store.fold_state(scenariod, "br:review-b")["rulings"]
    assert rulings_main["cand:seg:ch1/0004/0"] == "accept"
    assert rulings_alt["cand:seg:ch1/0004/1"] == "accept"


def test_counterexample_archive(scenariod):
    with open(os.path.join(PKG, "counterexamples.json"), encoding="utf-8") as fh:
        archive = json.load(fh)
    view = store.compute_branch_view(scenariod, CASE, "br:edge-rules")
    rules_by_seg = {}
    for e in view["rule_events"]:
        rules_by_seg.setdefault(e.get("segment"), set()).add(e["rule"])
    for case in archive["cases"]:
        if case["expect_rule"]:
            assert case["expect_rule"] in rules_by_seg.get(case["segment"], set()), \
                f"{case['id']} 缺少判定事件 {case['expect_rule']}"
    # 多峰段保留全部合法候选
    seg4 = next(s for s in view["segments"] if s["id"] == "seg:ch1/0004")
    assert len(seg4["candidates"]) >= 2
    # 区间版本修订只使测量失效（不影响对齐）
    meas_stale = store.invalidation_scope(scenariod, "br:main", "intervals:v1")
    assert meas_stale and all(s["kind"] == "meas" for s in meas_stale)
    align_stale = store.invalidation_scope(scenariod, "br:main", "params:v1")
    assert any(s["kind"] == "align" for s in align_stale)


def test_trace_back_from_final_state(scenariod):
    """从最终状态反查：哪条材料、哪版规则、哪次决定造成了差异。"""
    events = store.list_events(scenariod, "br:main")
    by_kind = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)
    patch = json.loads(by_kind["source-patch"][0]["payload"])
    assert patch["invalidated"]  # 失效范围随事件落档
    assert all("ch2/0005" in i["target"] for i in patch["invalidated"])
    rev = json.loads(by_kind["set-params"][0]["payload"])
    assert rev["params_version"] == "params:v2" and rev["invalidated"]
    # 每个派生项都能列出版本与输入来源
    drvs = store.stale_derivations(scenariod, "br:main")
    for d in drvs:
        inputs = json.loads(d["inputs"])
        assert any(i.startswith("src:") for i in inputs)
        assert any(i.startswith("params:") for i in inputs)
