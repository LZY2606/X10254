"""端到端（浏览器可见路径）：载入示例 -> 剧本 -> 分支裁定 -> 冲突 -> 重开验证历史。"""
import os

from app import create_app
from fastapi.testclient import TestClient


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "波形叠影台" in r.text


def test_full_browser_flow(tmp_path):
    db = str(tmp_path / "e2e.db")
    c = TestClient(create_app(db))

    # 1. 载入仓库内示例
    r = c.post("/api/cases/load")
    assert r.status_code == 200, r.text
    assert r.json()["case_id"] == "wave-overlay-demo"
    assert c.post("/api/cases/load").status_code == 409  # 重复载入被拒

    # 2. 推进剧本（导入基线 -> 参数修订 -> 区间修订 -> 局部源改动 -> 冲突决定 -> 边界分支）
    steps = []
    while True:
        r = c.post("/api/scenario/step").json()
        if r.get("done"):
            break
        steps.append(r)
    assert [s["step"] for s in steps] == [f"s{i}" for i in range(1, 8)]
    conflict_step = next(s for s in steps if s["step"] == "s6")
    assert conflict_step["result"]["conflict"]["their_events"]  # 冲突带双方差异

    # 3. 复核页面数据：叠影/抖动/候选/测量窗/排除原因，每个值带稳定身份
    view = c.get("/api/branches/br:main/view").json()
    assert view["versions"]["params_version"] == "params:v2"
    seg4 = next(s for s in view["segments"] if s["id"] == "seg:ch1/0004")
    assert len(seg4["candidates"]) >= 2
    assert all(c["id"].startswith("cand:seg:ch1/0004/") for c in seg4["candidates"])
    meas_ids = [m["id"] for per in view["measurements"].values() for m in per.values()]
    assert meas_ids and all(m.startswith("meas:") for m in meas_ids)
    assert any(e["segment"] == "seg:ch1/0007" for e in view["excluded"])

    # 4. 完成一次分支裁定 + 撤销
    head = view["head_seq"]
    r = c.post("/api/branches/br:main/events", json={
        "kind": "ruling", "base_seq": head,
        "payload": {"target": "cand:seg:ch1/0004/1", "decision": "reject"}})
    assert r.status_code == 200
    evt_id = r.json()["id"]
    v2 = c.get("/api/branches/br:main/view").json()
    assert v2["state"]["rulings"]["cand:seg:ch1/0004/1"] == "reject"
    r = c.post("/api/branches/br:main/undo",
               json={"event_id": evt_id, "base_seq": v2["head_seq"]})
    assert r.status_code == 200
    v3 = c.get("/api/branches/br:main/view").json()
    assert "cand:seg:ch1/0004/1" not in v3["state"]["rulings"]  # 恢复前一分支

    # 5. 制造一次冲突（两个“窗口”同时保存）
    head = v3["head_seq"]
    a = c.post("/api/branches/br:main/events", json={
        "kind": "ruling", "base_seq": head,
        "payload": {"target": "cand:seg:ch1/0004/0", "decision": "accept"}})
    assert a.status_code == 200
    b = c.post("/api/branches/br:main/events", json={
        "kind": "ruling", "base_seq": head,
        "payload": {"target": "cand:seg:ch1/0004/1", "decision": "accept"}})
    assert b.status_code == 409
    body = b.json()
    assert body["error"] == "version-conflict"
    assert body["their_events"] and body["your_event"]

    # 6. 时间旅行：回到基线事件，恢复当时依赖版本
    old = c.get("/api/branches/br:main/view?seq=1").json()
    assert old["versions"]["params_version"] == "params:v1"
    assert old["versions"]["interval_version"] == "intervals:v1"

    # 7. 导出 -> 删除本地数据库 -> 导入 -> 重新打开验证历史
    archive = c.get("/api/export").json()
    os.remove(db)
    c2 = TestClient(create_app(db))
    assert c2.get("/api/status").json()["case_id"] is None
    r = c2.post("/api/import", json=archive)
    assert r.status_code == 200
    st = c2.get("/api/status").json()
    assert st["case_id"] == "wave-overlay-demo"
    assert {b["id"] for b in st["branches"]} >= {"br:main", "br:review-b", "br:edge-rules"}
    again = c2.get("/api/branches/br:main/view").json()
    assert again["versions"] == view["versions"]
    assert again["jitter"] == view["jitter"]  # 身份/顺序/来源关系保持
    ev1 = [e["id"] for e in c.get("/api/branches/br:main/events").json()["events"]]
    ev2 = [e["id"] for e in c2.get("/api/branches/br:main/events").json()["events"]]
    assert ev1 == ev2
