"""波形叠影台 — 示波器分段、触发抖动与测量窗口谱系复核工具。

启动：.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5361
"""
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from waveforge import db as dbmod
from waveforge import exporter, ingest, scenario, store

BASE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE, "casepkg", "wave-overlay-demo")
STATIC = os.path.join(BASE, "static", "index.html")


def create_app(db_path=None, pkg_dir=PKG_DIR):
    db_path = db_path or os.environ.get("WAVEFORGE_DB",
                                        os.path.join(BASE, "waveforge.db"))
    app = FastAPI(title="波形叠影台")
    conn = dbmod.open_db(db_path)
    app.state.conn = conn
    app.state.pkg_dir = pkg_dir

    def case_id():
        row = conn.execute("SELECT DISTINCT case_id c FROM sources LIMIT 1").fetchone()
        return row["c"] if row else None

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(STATIC, encoding="utf-8") as fh:
            return fh.read()

    @app.get("/api/status")
    def status():
        cid = case_id()
        branches = [dict(r) for r in conn.execute(
            "SELECT * FROM branches ORDER BY created_gseq").fetchall()]
        for b in branches:
            b["head_seq"] = store.head_seq(conn, b["id"])
        return {"case_id": cid, "branches": branches,
                "scenario_cursor": scenario.scenario_cursor(conn, cid) if cid else 0,
                "sources": len(ingest.get_sources(conn))}

    @app.post("/api/cases/load")
    def load_case():
        try:
            info = ingest.load_case_package(conn, pkg_dir)
        except ingest.CaseAlreadyLoaded as exc:
            return JSONResponse({"error": f"case already loaded: {exc}"}, 409)
        except Exception as exc:  # 导入失败整体回滚，不留半份记录
            return JSONResponse({"error": f"import rolled back: {exc}"}, 400)
        return info

    @app.post("/api/scenario/step")
    def scenario_step():
        cid = case_id()
        if not cid:
            return JSONResponse({"error": "no case loaded"}, 400)
        try:
            result = scenario.apply_next_step(conn, pkg_dir, cid)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            return JSONResponse({"error": str(exc)}, 400)
        return result

    @app.get("/api/branches/{branch_id}/view")
    def branch_view(branch_id: str, seq: int = None):
        cid = case_id()
        if not cid:
            return JSONResponse({"error": "no case loaded"}, 400)
        view = store.compute_branch_view(conn, cid, branch_id, upto_seq=seq)
        conn.commit()
        view["branch_id"] = branch_id
        view["head_seq"] = store.head_seq(conn, branch_id)
        return view

    @app.get("/api/branches/{branch_id}/events")
    def branch_events(branch_id: str):
        return {"events": store.list_events(conn, branch_id)}

    @app.post("/api/branches/{branch_id}/events")
    async def post_event(branch_id: str, request: Request):
        body = await request.json()
        try:
            evt = store.append_event(conn, branch_id, body["kind"],
                                     body.get("payload", {}), body["base_seq"])
            conn.commit()
        except store.VersionConflict as exc:
            conn.rollback()
            return JSONResponse({
                "error": "version-conflict", "branch_id": exc.branch_id,
                "base_seq": exc.base_seq, "head_seq": exc.head_seq,
                "their_events": exc.their_events,
                "your_event": exc.your_event}, 409)
        return evt

    @app.post("/api/branches/{branch_id}/undo")
    async def undo(branch_id: str, request: Request):
        body = await request.json()
        try:
            evt = store.undo_event(conn, branch_id, body["event_id"], body["base_seq"])
            conn.commit()
        except store.VersionConflict as exc:
            conn.rollback()
            return JSONResponse({"error": "version-conflict",
                                 "head_seq": exc.head_seq}, 409)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, 404)
        return evt

    @app.post("/api/branches")
    async def new_branch(request: Request):
        body = await request.json()
        store.create_branch(conn, body["id"], body.get("name", body["id"]),
                            body.get("parent_id"), body.get("parent_seq"))
        conn.commit()
        return {"id": body["id"]}

    @app.get("/api/sources")
    def sources():
        cid = case_id()
        out = []
        for s in ingest.get_sources(conn, case_id=cid):
            out.append({k: s[k] for k in ("id", "kind", "name", "ord", "sha256")})
        return {"sources": out, "diagnostics": ingest.get_diagnostics(conn)}

    @app.get("/api/sources/{source_id:path}/raw")
    def source_raw(source_id: str):
        row = conn.execute("SELECT raw FROM sources WHERE id=?",
                           (source_id,)).fetchone()
        if row is None:
            return JSONResponse({"error": "not found"}, 404)
        return JSONResponse(content=dbmod.loads(row["raw"]))

    @app.get("/api/derivations")
    def derivations(branch_id: str):
        return {"derivations": store.stale_derivations(conn, branch_id)}

    @app.get("/api/export")
    def export():
        return exporter.export_archive(conn)

    @app.post("/api/import")
    async def import_(request: Request):
        archive = await request.json()
        try:
            counts = exporter.import_archive(conn, archive)
        except Exception as exc:
            return JSONResponse({"error": f"import rolled back: {exc}"}, 400)
        return {"imported": counts}

    return app


app = create_app()
