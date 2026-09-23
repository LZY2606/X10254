"""波形叠影台复核服务：FastAPI + SQLite + SVG 页面。

启动：.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5361
"""
from __future__ import annotations

import os

from fastapi import Body, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

import casepack
import state as state_mod
import store

DB_PATH = os.environ.get("WSG_DB", "waveform.db")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "static", "page.html")


def _conflict_response(exc):
    return JSONResponse(status_code=409, content=exc.to_dict())


def create_app(db_path=DB_PATH):
    conn = store.connect(db_path)
    store.init_db(conn)
    app = FastAPI(title="波形叠影台")

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(PAGE, "r", encoding="utf-8") as fh:
            return fh.read()

    @app.post("/api/case/load")
    def load_case():
        if store.has_sources(conn):
            return {"status": "already-loaded"}
        data = casepack.load_baseline()
        store.import_sources(conn, casepack.baseline_sources(data))
        store.create_branch(conn, "br-main", "main")
        store.add_event(conn, "br-main", "import", {
            "case_id": data["case_id"],
            "segments": [s["segment_id"] for s in data["segments"]],
            "note": "初始导入：每段原采样、时基、竖直增益、探头衰减、"
                    "触发时标、deskew 与测量区间版本化",
        })
        state_mod.refresh_derived(conn, "br-main")
        return {"status": "loaded", "case_id": data["case_id"]}

    @app.get("/api/state")
    def get_state(branch: str = Query("br-main"), upto: int = Query(None)):
        return state_mod.compute_state(conn, branch, upto)

    @app.get("/api/archive")
    def get_archive():
        return {"counterpoints": casepack.load_counterpoints(),
                "revisions": casepack.load_revisions()}

    @app.post("/api/branches")
    def create_branch(body: dict = Body(...)):
        parent = body.get("parent", "br-main")
        existing = {b["branch_id"] for b in store.list_branches(conn)}
        idx = 1
        while "br-%02d" % idx in existing or "br-main" == "br-%02d" % idx:
            idx += 1
        branch_id = body.get("branch_id") or "br-%02d" % idx
        br = store.create_branch(conn, branch_id,
                                 body.get("name", branch_id), parent)
        state_mod.refresh_derived(conn, branch_id)
        return br

    @app.post("/api/rulings")
    def post_ruling(body: dict = Body(...)):
        branch = body.get("branch", "br-main")
        payload = {"segment_id": body["segment_id"],
                   "candidate_id": body["candidate_id"],
                   "decision": body["decision"],
                   "reviewer": body.get("reviewer", "reviewer")}
        try:
            ev = store.add_event(conn, branch, "ruling", payload,
                                 expected_head=body.get("expected_head"))
        except store.Conflict as exc:
            return _conflict_response(exc)
        state_mod.refresh_derived(conn, branch)
        return {"ok": True, "event": ev}

    @app.post("/api/undo")
    def post_undo(body: dict = Body(...)):
        branch = body.get("branch", "br-main")
        payload = {"target_seq": body["target_seq"]}
        try:
            ev = store.add_event(conn, branch, "undo", payload,
                                 expected_head=body.get("expected_head"))
        except store.Conflict as exc:
            return _conflict_response(exc)
        state_mod.refresh_derived(conn, branch)
        return {"ok": True, "event": ev}

    @app.post("/api/restore")
    def post_restore(body: dict = Body(...)):
        branch = body.get("branch", "br-main")
        payload = {"to_seq": body["to_seq"]}
        try:
            ev = store.add_event(conn, branch, "restore", payload,
                                 expected_head=body.get("expected_head"))
        except store.Conflict as exc:
            return _conflict_response(exc)
        state_mod.refresh_derived(conn, branch)
        return {"ok": True, "event": ev}

    @app.post("/api/revisions/params")
    def post_param_revision(body: dict = Body(...)):
        branch = body.get("branch", "br-main")
        rev = casepack.load_revisions()["param_revision"]
        current = state_mod.compute_state(conn, branch)
        affected = state_mod.invalidation_scope(
            current, changed_params=[rev["from"]])
        payload = dict(rev)
        payload["affected"] = affected
        try:
            ev = store.add_event(conn, branch, "param-revision", payload,
                                 expected_head=body.get("expected_head"))
        except store.Conflict as exc:
            return _conflict_response(exc)
        state_mod.refresh_derived(conn, branch)
        return {"ok": True, "event": ev, "affected": affected}

    @app.post("/api/revisions/source")
    def post_source_revision(body: dict = Body(...)):
        branch = body.get("branch", "br-main")
        change = next(c for c in casepack.load_revisions()["source_changes"]
                      if c["source_id"] == body["source_id"])
        current = state_mod.compute_state(conn, branch)
        affected = state_mod.invalidation_scope(
            current, changed_sources=[change["source_id"]])
        try:
            with conn:
                store.put_source_version(conn, change["source_id"],
                                         change["version"], "segment",
                                         change["payload"])
                payload = {k: change[k] for k in
                           ("change_id", "source_id", "version", "note")}
                payload["affected"] = affected
                ev = store.add_event(conn, branch, "source-change", payload,
                                     expected_head=body.get("expected_head"))
        except store.Conflict as exc:
            return _conflict_response(exc)
        state_mod.refresh_derived(conn, branch)
        return {"ok": True, "event": ev, "affected": affected}

    @app.get("/api/export")
    def export():
        return store.export_db(conn)

    @app.post("/api/import")
    def import_(data: dict = Body(...)):
        try:
            store.import_db(conn, data)
        except store.Conflict as exc:
            return _conflict_response(exc)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return {"ok": True}

    return app


app = create_app()
