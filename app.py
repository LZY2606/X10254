"""FastAPI application: waveform overlay / segmented alignment / jitter review.

Start: ``.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 5361``
"""
from __future__ import annotations

import json
import os
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from appcore import svg as SVG
from appcore.casegen import build_case
from appcore.ingest import bundled_case, ingest_case, validate_case
from appcore.projection import project
from appcore.repository import Conflict, Repository
from appcore import services as SVC

ROOT = pathlib.Path(__file__).resolve().parent
DB_PATH = os.environ.get("WFLINE_DB", str(ROOT / "data" / "review.sqlite3"))
BUNDLED_CASE = ROOT / "cases" / "waveform-lineage" / "case.json"

app = FastAPI(title="waveform-lineage review")


def get_repo() -> Repository:
    if not getattr(app.state, "repo", None):
        pathlib.Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        app.state.repo = Repository(DB_PATH)
        app.state.case = None
    return app.state.repo


def case_loaded() -> dict:
    repo = get_repo()
    if getattr(app.state, "case", None) is None:
        raise HTTPException(409, "no case loaded; POST /api/load-bundled first")
    return app.state.case


def state_for(branch_id: str) -> dict:
    repo = get_repo()
    case = case_loaded()
    return project(repo.branch_events(branch_id), case)


# ---------------------------------------------------------------------- state
@app.get("/api/state")
def get_state(branch: str = "main"):
    state = state_for(branch)
    repo = get_repo()
    events = repo.branch_events(branch)
    state["branch"] = branch
    state["tip_event"] = events[-1]["event_id"] if events else None
    state["branches"] = repo.list_branches()
    return state


@app.get("/api/case")
def get_case():
    return case_loaded()


@app.post("/api/load-bundled")
def load_bundled():
    repo = get_repo()
    if repo.all_events():
        raise HTTPException(409, "repository already contains a case")
    case = json.loads(BUNDLED_CASE.read_text()) if BUNDLED_CASE.exists() \
        else build_case()
    result = ingest_case(repo, case)
    app.state.case = case
    return {"loaded": True, "case_id": case["case_id"],
            "event": result["event"], "material_count": len(result["materials"])}


# ------------------------------------------------------------------ mutations
class WindowReq(BaseModel):
    to_version: str
    branch: str = "main"
    expected_last_event: str | None = None


@app.post("/api/window")
def post_window(req: WindowReq):
    repo = get_repo()
    try:
        e = SVC.revise_window(
            repo, to_version=req.to_version, branch_id=req.branch,
            expected_last_event=req.expected_last_event)
    except Conflict as c:
        return JSONResponse(status_code=409, content={
            "error": "version-conflict", "message": c.message,
            "server_event_id": c.server_event_id,
            "your_event_id": c.your_event_id, "diff": c.diff})
    return {"event": e}


class PatchReq(BaseModel):
    segment: str
    channel: str
    field: str
    value: object
    branch: str = "main"
    expected_last_event: str | None = None


@app.post("/api/source-patch")
def post_patch(req: PatchReq):
    repo = get_repo()
    try:
        e = SVC.patch_source(
            repo, segment=req.segment, channel=req.channel,
            field=req.field, change={req.field: req.value},
            branch_id=req.branch,
            expected_last_event=req.expected_last_event)
    except Conflict as c:
        return JSONResponse(status_code=409, content={
            "error": "version-conflict", "diff": c.diff})
    return {"event": e,
            "invalidation": e["payload"]["invalidation"]
            if isinstance(e["payload"], dict) else None}


class RulingReq(BaseModel):
    segment: str
    candidate_id: str
    decision: str  # "accept" | "reject"
    note: str = ""
    branch: str = "main"
    expected_last_event: str | None = None


@app.post("/api/rule")
def post_rule(req: RulingReq):
    repo = get_repo()
    try:
        e = SVC.rule_candidate(
            repo, segment=req.segment, candidate_id=req.candidate_id,
            decision=req.decision, note=req.note, branch_id=req.branch,
            expected_last_event=req.expected_last_event)
    except Conflict as c:
        return JSONResponse(status_code=409, content={
            "error": "version-conflict", "diff": c.diff})
    return {"event": e}


class UndoReq(BaseModel):
    event_id: str
    branch: str = "main"
    label: str = "undo"


@app.post("/api/undo")
def post_undo(req: UndoReq):
    repo = get_repo()
    try:
        return SVC.undo_to_before_event(
            repo, event_id=req.event_id, branch_id=req.branch, label=req.label)
    except ValueError as e:
        raise HTTPException(404, str(e))


class ForkReq(BaseModel):
    base_branch: str = "main"
    label: str = "reinterpretation"
    at_event: str | None = None


@app.post("/api/branch")
def post_branch(req: ForkReq):
    return SVC.fork_branch(get_repo(), base_branch=req.base_branch,
                           label=req.label, at_event=req.at_event)


@app.get("/api/events")
def get_events(branch: str = "main"):
    repo = get_repo()
    return {"branch": branch, "events": repo.branch_events(branch)}


# --------------------------------------------------------------------- export
@app.get("/api/export")
def export_bundle():
    return get_repo().export_bundle()


@app.post("/api/import")
def import_bundle(bundle: dict):
    repo = get_repo()
    try:
        n = repo.import_bundle(bundle)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # case data is not stored in bundle events; use bundled case for projection
    app.state.case = json.loads(BUNDLED_CASE.read_text())
    return {"imported": n}


@app.post("/api/reset")
def reset_db():
    repo = get_repo()
    repo.close()
    app.state.repo = None
    app.state.case = None
    if DB_PATH != ":memory:" and pathlib.Path(DB_PATH).exists():
        pathlib.Path(DB_PATH).unlink()
    return {"reset": True}


# ------------------------------------------------------------------------ svg
@app.get("/svg/overlay", response_class=Response)
def svg_overlay(branch: str = "main", segment: str | None = None):
    state = state_for(branch)
    return Response(SVG.overlay_svg(case_loaded(), state,
                                    show_segment=segment),
                    media_type="image/svg+xml")


@app.get("/svg/jitter", response_class=Response)
def svg_jitter(branch: str = "main"):
    return Response(SVG.jitter_svg(state_for(branch)),
                    media_type="image/svg+xml")


@app.get("/svg/segment/{seg_id}", response_class=Response)
def svg_segment(seg_id: str, branch: str = "main"):
    return Response(SVG.segment_svg(case_loaded(), state_for(branch), seg_id),
                    media_type="image/svg+xml")


app.mount("/static", StaticFiles(directory=str(ROOT / "web")), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text()
