"""案例剧本回放：把“波形叠影台”案例的叙事步骤变成事件流。

步骤（与 casepkg 的 scenario.json 对应）：
  1 import-baseline   初始导入基线（每段原采样/时基/增益/衰减/触发/deskew/区间版本）
  2 review-baseline   审阅者核对基线
  3 param-revision    参数修订 -> 依赖失效范围
  4 source-patch      只影响局部依赖的源数据改动（ch2 deskew 标定修订）
  5 conflicting-decisions 互相冲突的人工决定（后者遇版本冲突，落到新分支，两者都保留）
  6 edge-branch       第二分支检验 R1/R2/R3 规则并保留全部合法候选
"""
import json
import os

from . import store


def load_scenario(pkg_dir):
    with open(os.path.join(pkg_dir, "scenario.json"), encoding="utf-8") as fh:
        return json.load(fh)


def scenario_cursor(conn, case_id):
    row = conn.execute(
        "SELECT value FROM meta WHERE key=?", (f"scenario_cursor:{case_id}",)).fetchone()
    return int(row["value"]) if row else 0


def _set_cursor(conn, case_id, step):
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                 (f"scenario_cursor:{case_id}", str(step)))


def apply_next_step(conn, pkg_dir, case_id):
    scenario = load_scenario(pkg_dir)
    steps = scenario["steps"]
    cursor = scenario_cursor(conn, case_id)
    if cursor >= len(steps):
        return {"done": True, "cursor": cursor}
    step = steps[cursor]
    result = _apply_step(conn, case_id, step)
    _set_cursor(conn, case_id, cursor + 1)
    return {"done": False, "cursor": cursor + 1, "step": step["id"],
            "title": step.get("title"), "result": result}


def _apply_step(conn, case_id, step):
    kind = step["kind"]
    main = "br:main"
    if kind == "import-baseline":
        store.create_branch(conn, main, "主线复核")
        versions = {"rule_version": "align-rules:v1", "params_version": "params:v1",
                    "interval_version": "intervals:v1"}
        evt = store.append_event(conn, main, "import-baseline",
                                 {"case_id": case_id, "interval_version": "intervals:v1"},
                                 base_seq=0, versions=versions)
        store.append_event(conn, main, "set-intervals",
                           {"interval_version": "intervals:v1"}, base_seq=evt["seq"])
        return {"branch": main}
    if kind == "review-baseline":
        head = store.head_seq(conn, main)
        store.append_event(conn, main, "note",
                           {"text": step["note"]}, base_seq=head)
        return {"noted": True}
    if kind == "param-revision":
        head = store.head_seq(conn, main)
        store.compute_branch_view(conn, case_id, main)  # 确保基线派生已记录
        evt = store.append_event(conn, main, "set-params",
                                 {"params_version": step["params_version"]},
                                 base_seq=head)
        store.compute_branch_view(conn, case_id, main)
        scope = store.invalidation_scope(conn, main, "params:v1")
        payload = {"params_version": step["params_version"], "invalidated": scope}
        conn.execute("UPDATE events SET payload=? WHERE id=?",
                     (store.dumps(payload), evt["id"]))
        return {"invalidated": scope}
    if kind == "source-patch":
        head = store.head_seq(conn, main)
        store.compute_branch_view(conn, case_id, main)
        evt = store.append_event(conn, main, "source-patch",
                                 {"source_id": step["source_id"],
                                  "new_source_id": step["new_source_id"]},
                                 base_seq=head)
        store.compute_branch_view(conn, case_id, main)
        scope = store.invalidation_scope(conn, main, step["source_id"])
        payload = {"source_id": step["source_id"],
                   "new_source_id": step["new_source_id"], "invalidated": scope}
        conn.execute("UPDATE events SET payload=? WHERE id=?",
                     (store.dumps(payload), evt["id"]))
        return {"invalidated": scope}
    if kind == "interval-revision":
        head = store.head_seq(conn, main)
        store.compute_branch_view(conn, case_id, main)
        evt = store.append_event(conn, main, "set-intervals",
                                 {"interval_version": step["interval_version"]},
                                 base_seq=head)
        store.compute_branch_view(conn, case_id, main)
        scope = store.invalidation_scope(conn, main, "intervals:v1")
        payload = {"interval_version": step["interval_version"], "invalidated": scope}
        conn.execute("UPDATE events SET payload=? WHERE id=?",
                     (store.dumps(payload), evt["id"]))
        return {"invalidated": scope}
    if kind == "conflicting-decisions":
        head = store.head_seq(conn, main)
        d1, d2 = step["decisions"]
        store.append_event(conn, main, "ruling", d1, base_seq=head)
        # 第二个决定来自过期 base_seq（模拟另一窗口同时保存）-> 版本冲突
        conflict = None
        try:
            store.append_event(conn, main, "ruling", d2, base_seq=head)
        except store.VersionConflict as exc:
            conflict = {"base_seq": exc.base_seq, "head_seq": exc.head_seq,
                        "their_events": exc.their_events, "your_event": exc.your_event}
        # 冲突决定落到新分支，采纳与未采纳的结果都保留
        store.create_branch(conn, "br:review-b", "复核B（冲突决定）",
                            parent_id=main, parent_seq=head)
        store.append_event(conn, "br:review-b", "ruling", d2, base_seq=0)
        return {"conflict": conflict, "branch": "br:review-b"}
    if kind == "edge-branch":
        head = store.head_seq(conn, main)
        store.create_branch(conn, "br:edge-rules", "边界规则检验",
                            parent_id=main, parent_seq=1)
        store.append_event(conn, "br:edge-rules", "note",
                           {"text": step["note"]}, base_seq=0)
        view = store.compute_branch_view(conn, case_id, "br:edge-rules")
        return {"rule_events": len(view["rule_events"])}
    raise ValueError(f"unknown scenario step kind: {kind}")
