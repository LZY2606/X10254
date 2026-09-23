"""Waveform overlay review tool — segmented capture alignment, trigger
jitter and measurement-window provenance. FastAPI + SQLite + NumPy + SVG.
"""
import json
import os

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

import core
import db
import derive

CASE_PATH = os.path.join(os.path.dirname(__file__), "casepack", "case.json")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def load_case_file():
    with open(CASE_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_case(case):
    if case.get("format") != "waveform-overlay-case/1":
        raise ValueError("unsupported case format: %r" % case.get("format"))
    for s in case.get("segments", []):
        for field in ("seg_key", "channel", "t", "v", "trigger_ts",
                      "vertical_gain", "probe_atten", "deskew",
                      "meas_window_version"):
            if field not in s:
                raise ValueError("segment %r missing field %r"
                                 % (s.get("seg_key"), field))
        tv = s["t"].get("values", [])
        vv = s["v"].get("values", [])
        if not tv or len(tv) != len(vv):
            raise ValueError("segment %r: t/v length mismatch" % s.get("seg_key"))


def import_case(conn, case):
    """Atomic import: any failure rolls back the whole change — no half
    records. Raw representations are stored verbatim; normalization only
    happens in the derived layer."""
    validate_case(case)
    imp_id = db.new_id("imp")
    try:
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO imports(id,name,created_at,status) VALUES(?,?,?,?)",
            (imp_id, case["name"], db.now(), "ok"))
        seg_ids = []
        for i, s in enumerate(case["segments"]):
            sid = db.new_id("seg")
            norm, diags = derive.normalize_segment(s)
            conn.execute(
                "INSERT INTO segments(id,import_id,seg_key,channel,ordinal,"
                "raw_json,source_hash,diagnostics) VALUES(?,?,?,?,?,?,?,?)",
                (sid, imp_id, s["seg_key"], s["channel"], i,
                 json.dumps(s, ensure_ascii=False), derive.source_hash(s),
                 json.dumps(diags, ensure_ascii=False)))
            seg_ids.append(sid)
        db.set_meta(conn, "case_meta", {
            "params": case["params"],
            "meas_windows": case["meas_windows"],
            "initial_mw_version": case["initial_mw_version"],
            "name": case["name"],
            "narrative": case["narrative"],
        })
        br = conn.execute("SELECT id FROM branches WHERE name='main'").fetchone()
        if br is None:
            bid = db.new_id("br")
            conn.execute(
                "INSERT INTO branches(id,name,parent_id,base_event_id,"
                "head_event_id,created_seq) VALUES(?,?,NULL,NULL,NULL,0)",
                (bid, "main"))
        else:
            bid = br["id"]
        db.append_event(conn, bid, "import",
                        {"import_id": imp_id, "segment_ids": seg_ids,
                         "name": case["name"]})
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return imp_id, bid


def branch_view(conn, branch_id, upto_event_id=None):
    state, chain, case_meta = core.replay(conn, branch_id, upto_event_id)
    derived = derive.compute_derived(state, case_meta)
    br = conn.execute("SELECT * FROM branches WHERE id=?",
                      (branch_id,)).fetchone()
    segments = []
    for sid in state["order"]:
        seg = state["segments"][sid]
        norm = derived["normalized"].get(sid)
        segments.append({
            "id": sid, "key": seg["key"], "channel": seg["channel"],
            "hash": seg["hash"],
            "raw": seg["raw"],
            "diagnostics": seg["diagnostics"],
            "normalized": norm,
        })
    return {
        "branch": {"id": br["id"], "name": br["name"],
                   "parent_id": br["parent_id"],
                   "head_event_id": br["head_event_id"]},
        "history_view": upto_event_id is not None,
        "upto_event_id": upto_event_id,
        "case": {"name": case_meta.get("name"),
                 "narrative": case_meta.get("narrative")},
        "params": state["params"],
        "mw_version": state["mw_version"],
        "meas_windows": case_meta.get("meas_windows", {}),
        "segments": segments,
        "derived": {k: v for k, v in derived.items() if k != "normalized"},
        "events": [{"id": e["id"], "seq": e["seq"], "kind": e["kind"],
                    "payload": e["payload"], "created_at": e["created_at"]}
                   for e in chain],
        "rules": [
            {"id": "R1", "text": "时间轴不等间隔时不用索引相减"},
            {"id": "R2", "text": "缺样与饱和区间不进入插值"},
            {"id": "R3", "text": "对齐保留多峰互相关候选"},
        ],
    }


def create_app(db_path="app.db"):
    app = FastAPI(title="波形叠影台")
    conn = db.connect(db_path)

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/state")
    def get_state(branch: str = Query("main"), upto: str = Query(None)):
        row = conn.execute("SELECT id FROM branches WHERE name=? OR id=?",
                           (branch, branch)).fetchone()
        if row is None:
            if not conn.execute("SELECT 1 FROM imports LIMIT 1").fetchone():
                imp_id, bid = import_case(conn, load_case_file())
                conn.commit()
                core.refresh_derived(conn, bid)
                row = conn.execute("SELECT id FROM branches WHERE id=?",
                                   (bid,)).fetchone()
            else:
                raise HTTPException(404, "branch not found: %s" % branch)
        try:
            return branch_view(conn, row["id"], upto)
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/branches")
    def list_branches():
        rows = conn.execute(
            "SELECT * FROM branches ORDER BY created_seq, rowid").fetchall()
        return {"branches": [dict(r) for r in rows]}

    @app.post("/api/branches")
    def create_branch(body: dict = Body(...)):
        name = body.get("name")
        base_branch = body.get("base_branch", "main")
        base_event_id = body.get("base_event_id")
        parent = conn.execute(
            "SELECT * FROM branches WHERE name=? OR id=?",
            (base_branch, base_branch)).fetchone()
        if parent is None:
            raise HTTPException(404, "base branch not found")
        if base_event_id is None:
            base_event_id = parent["head_event_id"]
        bid = db.new_id("br")
        seq = conn.execute("SELECT COALESCE(MAX(created_seq),0)+1 AS s "
                           "FROM branches").fetchone()["s"]
        conn.execute(
            "INSERT INTO branches(id,name,parent_id,base_event_id,"
            "head_event_id,created_seq) VALUES(?,?,?,?,NULL,?)",
            (bid, name, parent["id"], base_event_id, seq))
        conn.commit()
        core.refresh_derived(conn, bid)
        return {"id": bid, "name": name, "base_event_id": base_event_id}

    @app.post("/api/branches/{branch_id}/events")
    def post_event(branch_id: str, body: dict = Body(...)):
        kind = body.get("kind")
        payload = body.get("payload", {})
        expected_head = body.get("expected_head", "__check__")
        br = conn.execute("SELECT id FROM branches WHERE name=? OR id=?",
                          (branch_id, branch_id)).fetchone()
        if br is None:
            raise HTTPException(404, "branch not found")
        if kind == "adjudicate" and payload.get("decision") not in ("accept", "reject"):
            raise HTTPException(400, "decision must be accept|reject")
        try:
            eid = db.append_event(conn, br["id"], kind, payload, expected_head)
            conn.commit()
        except db.ConflictError as c:
            conn.rollback()
            return JSONResponse(status_code=409, content={
                "error": "version_conflict",
                "message": "分支头部已移动：后提交者需基于新头部重放",
                "current_head": c.current_head,
                "their_events": c.their_events,
                "your_event": {"kind": kind, "payload": payload},
            })
        except KeyError as e:
            raise HTTPException(404, str(e))
        core.refresh_derived(conn, br["id"])
        return {"event_id": eid, "head": eid}

    @app.get("/api/branches/{branch_id}/invalidations")
    def invalidations(branch_id: str):
        br = conn.execute("SELECT id FROM branches WHERE name=? OR id=?",
                          (branch_id, branch_id)).fetchone()
        if br is None:
            raise HTTPException(404, "branch not found")
        return {"invalidated": core.invalidation_scope(conn, br["id"])}

    @app.get("/api/export")
    def export():
        out = {"format": "waveform-overlay-export/1", "tables": {}}
        for table in db.TABLES:
            rows = conn.execute("SELECT * FROM %s" % table).fetchall()
            out["tables"][table] = [dict(r) for r in rows]
        return out

    @app.post("/api/import")
    def import_dump(body: dict = Body(...)):
        if body.get("format") != "waveform-overlay-export/1":
            raise HTTPException(400, "unsupported export format")
        try:
            conn.execute("BEGIN")
            for table in db.TABLES:
                conn.execute("DELETE FROM %s" % table)
            for table in db.TABLES:
                for row in body["tables"].get(table, []):
                    cols = list(row.keys())
                    conn.execute(
                        "INSERT INTO %s(%s) VALUES(%s)"
                        % (table, ",".join(cols), ",".join("?" * len(cols))),
                        [row[c] for c in cols])
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            raise HTTPException(400, "import failed, rolled back: %s" % e)
        return {"ok": True,
                "counts": {t: len(body["tables"].get(t, [])) for t in db.TABLES}}

    @app.post("/api/reset")
    def reset():
        for table in db.TABLES:
            conn.execute("DELETE FROM %s" % table)
        conn.commit()
        return {"ok": True}

    return app


app = create_app(os.environ.get("APP_DB", "app.db"))
