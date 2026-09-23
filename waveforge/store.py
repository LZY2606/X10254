"""事件日志 / 分支 / 裁定 / 失效范围。

- 视图选择、候选切换、接受/否定都只追加事件，不改原数据。
- 裁定可撤销（undo 事件），折叠时跳过被撤销事件，恢复前一分支状态。
- 两个窗口同时保存：base_seq 过期 -> VersionConflict，带双方差异。
- 参数/参考版本变化后按依赖关系计算失效范围，不全局替换旧结论。
"""
import hashlib

from . import db as dbmod
from . import derive
from .db import dumps, loads

DEFAULT_PARAMS = {
    "v1": {"version": "params:v1", "max_lag_s": 15e-9, "peak_threshold": 0.8},
    "v2": {"version": "params:v2", "max_lag_s": 15e-9, "peak_threshold": 0.9},
}


class VersionConflict(Exception):
    def __init__(self, branch_id, base_seq, head_seq, their_events, your_event):
        self.branch_id = branch_id
        self.base_seq = base_seq
        self.head_seq = head_seq
        self.their_events = their_events
        self.your_event = your_event
        super().__init__(f"version-conflict on {branch_id}: base={base_seq} head={head_seq}")


# ---------------------------------------------------------------- 分支与事件
def create_branch(conn, branch_id, name, parent_id=None, parent_seq=None):
    gseq = _next_gseq_hint(conn)
    conn.execute(
        "INSERT INTO branches(id, name, parent_id, parent_seq, created_gseq)"
        " VALUES(?,?,?,?,?)", (branch_id, name, parent_id, parent_seq, gseq))
    return branch_id


def _next_gseq_hint(conn):
    row = conn.execute("SELECT MAX(gseq) g FROM events").fetchone()
    return (row["g"] or 0) + 1


def head_seq(conn, branch_id):
    row = conn.execute(
        "SELECT MAX(seq) s FROM events WHERE branch_id=?", (branch_id,)).fetchone()
    return row["s"] or 0


def list_events(conn, branch_id, since_seq=0, upto_seq=None):
    sql = "SELECT * FROM events WHERE branch_id=? AND seq>?"
    args = [branch_id, since_seq]
    if upto_seq is not None:
        sql += " AND seq<=?"
        args.append(upto_seq)
    sql += " ORDER BY seq"
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def append_event(conn, branch_id, kind, payload, base_seq, versions=None):
    """追加事件；base_seq 过期抛 VersionConflict（携带双方差异）。"""
    head = head_seq(conn, branch_id)
    if base_seq != head:
        their = [ {"id": e["id"], "seq": e["seq"], "kind": e["kind"],
                   "payload": loads(e["payload"])} for e in list_events(conn, branch_id, base_seq)]
        raise VersionConflict(branch_id, base_seq, head, their,
                              {"kind": kind, "payload": payload})
    seq = head + 1
    evt_id = f"evt:{branch_id}/{seq:04d}"
    versions = versions or current_versions(conn, branch_id)
    conn.execute(
        "INSERT INTO events(id, branch_id, seq, kind, payload, rule_version,"
        " params_version, interval_version, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (evt_id, branch_id, seq, kind, dumps(payload),
         versions.get("rule_version"), versions.get("params_version"),
         versions.get("interval_version"), dbmod.now()))
    return {"id": evt_id, "seq": seq, "kind": kind, "payload": payload}


def undo_event(conn, branch_id, target_event_id, base_seq):
    row = conn.execute(
        "SELECT * FROM events WHERE id=? AND branch_id=?",
        (target_event_id, branch_id)).fetchone()
    if row is None:
        raise KeyError(f"unknown event {target_event_id}")
    evt = append_event(conn, branch_id, "undo", {"target": target_event_id}, base_seq)
    conn.execute("UPDATE events SET undone=1 WHERE id=?", (target_event_id,))
    return evt


# ---------------------------------------------------------------- 状态折叠
def fold_state(conn, branch_id, upto_seq=None):
    """折叠分支事件 -> 当前版本组合 + 生效裁定。回到任意旧事件即恢复当时版本。"""
    events = list_events(conn, branch_id, 0, upto_seq)
    undone = {e["id"] for e in events if e["undone"]}
    state = {"rule_version": derive.RULE_VERSION, "params_version": "params:v1",
             "interval_version": None, "rulings": {}, "sources_override": {},
             "events": [], "notes": []}
    for e in events:
        if e["id"] in undone and e["kind"] != "undo":
            continue
        p = loads(e["payload"])
        state["events"].append({"id": e["id"], "seq": e["seq"], "kind": e["kind"],
                                "payload": p, "undone": e["id"] in undone})
        if e["kind"] == "set-params":
            state["params_version"] = p["params_version"]
        elif e["kind"] == "set-intervals":
            state["interval_version"] = p["interval_version"]
        elif e["kind"] == "source-patch":
            state["sources_override"][p["source_id"]] = p["new_source_id"]
        elif e["kind"] == "ruling":
            state["rulings"][p["target"]] = p["decision"]
        elif e["kind"] == "undo":
            pass
    return state


def current_versions(conn, branch_id):
    s = fold_state(conn, branch_id)
    return {"rule_version": s["rule_version"], "params_version": s["params_version"],
            "interval_version": s["interval_version"]}


# ---------------------------------------------------------------- 派生视图
def _params_obj(state):
    return DEFAULT_PARAMS[state["params_version"].split(":")[-1]]


def _load_materials(conn, case_id, state):
    """按状态取来源（含 source-patch 覆盖），返回段 raw 列表与区间对象。"""
    from .ingest import get_sources
    srcs = get_sources(conn, case_id=case_id)
    by_id = {s["id"]: s for s in srcs}
    override = state["sources_override"]
    replaced = set(override.values())
    segs, intervals = [], {"version": "intervals:none", "windows": []}
    for s in srcs:
        if s["kind"] == "segment":
            if s["id"] in replaced:
                continue  # 被 source-patch 替换的来源由覆盖关系引入
            src = by_id.get(override.get(s["id"], s["id"]), s)
            segs.append({"source_id": src["id"], "raw": dbmod.loads(src["raw"])})
        elif s["kind"] == "intervals":
            if state["interval_version"] and s["name"].endswith(state["interval_version"]):
                intervals = dbmod.loads(s["raw"])
            elif state["interval_version"] is None and "v1" in s["name"]:
                intervals = dbmod.loads(s["raw"])
    return segs, intervals


def compute_branch_view(conn, case_id, branch_id, upto_seq=None):
    """计算分支可见结论；派生项写入 derivations 缓存并按依赖标记失效。"""
    state = fold_state(conn, branch_id, upto_seq)
    segs_mat, intervals = _load_materials(conn, case_id, state)
    views = []
    for m in segs_mat:
        v = derive.build_segment_view(m["raw"])
        v["source_id"] = m["source_id"]
        views.append(v)
    views.sort(key=lambda v: v["id"])
    params = _params_obj(state)
    alignments = derive.align_segments(views, params)
    view = derive.compute_view(views, alignments, state["rulings"], intervals,
                               views[0]["id"])
    view["versions"] = {"rule_version": state["rule_version"],
                        "params_version": state["params_version"],
                        "interval_version": intervals.get("version")}
    view["state"] = {"rulings": state["rulings"], "events": state["events"]}
    _record_derivations(conn, branch_id, state, intervals, views, alignments, view)
    return view


def _drv_id(kind, target, versions, inputs):
    h = hashlib.sha256(dumps({"k": kind, "t": target, "v": versions,
                              "i": inputs}).encode()).hexdigest()[:16]
    return f"drv:{kind}:{target}:{h}"


def _record_derivations(conn, branch_id, state, intervals, views, alignments, view):
    versions = {"rule": state["rule_version"], "params": state["params_version"],
                "interval": intervals.get("version")}
    items = []
    for v in views:
        inputs = [v["source_id"], state["params_version"], state["rule_version"]]
        items.append(("align", v["id"], inputs,
                      {"candidates": alignments[v["id"]]["candidates"]}))
        for wid, meas in view["measurements"].get(v["id"], {}).items():
            items.append(("meas", f"{v['id']}/{wid}",
                          inputs + [intervals.get("version")], meas))
    current_ids = set()
    gseq = _next_gseq_hint(conn)
    for kind, target, inputs, output in items:
        did = _drv_id(kind, target, versions, inputs)
        current_ids.add(did)
        row = conn.execute("SELECT id FROM derivations WHERE id=?", (did,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO derivations(id, branch_id, kind, target, versions,"
                " inputs, output, stale, created_gseq) VALUES(?,?,?,?,?,?,?,0,?)",
                (did, branch_id, kind, target, dumps(versions), dumps(inputs),
                 dumps(output), gseq))
    # 失效范围：同 (kind,target) 下不再被当前状态复现的旧派生项标记 stale
    rows = conn.execute(
        "SELECT id, kind, target FROM derivations WHERE branch_id=?",
        (branch_id,)).fetchall()
    for r in rows:
        stale = 0 if r["id"] in current_ids else 1
        conn.execute("UPDATE derivations SET stale=? WHERE id=?", (stale, r["id"]))


def invalidation_scope(conn, branch_id, ref):
    """依赖 ref 的失效派生项（不允许全局悄悄替换旧结论，范围必须显式）。"""
    rows = conn.execute(
        "SELECT id, kind, target, inputs FROM derivations"
        " WHERE branch_id=? AND stale=1", (branch_id,)).fetchall()
    return [{"id": r["id"], "kind": r["kind"], "target": r["target"]}
            for r in rows if ref in loads(r["inputs"])]


def stale_derivations(conn, branch_id):
    rows = conn.execute(
        "SELECT id, kind, target, inputs, stale FROM derivations WHERE branch_id=?"
        " ORDER BY id", (branch_id,)).fetchall()
    return [dict(r) for r in rows]
