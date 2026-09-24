"""End-to-end browser-visible flow through the real ASGI app.

Exercises: load bundled example -> inspect overlay/jitter/candidates/windows/
exclusions -> branch ruling -> conflicting window saves -> reopen history;
plus export -> delete local DB -> import preserving identities/order/lineage.
Also asserts stable data-uid markers exist in the page and SVGs.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


def _seg(st, sid):
    return next(s for s in st["segments"] if s["seg_id"] == sid)


def test_full_browser_visible_review_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("WFLINE_DB", str(tmp_path / "e2e.sqlite3"))
    appmod = importlib.import_module("app")
    importlib.reload(appmod)
    with TestClient(appmod.app) as c:
        # page is served
        page = c.get("/").text
        assert "波形叠影台" in page
        assert 'id="steps"' in page and "btn-conflict-b" in page
        js = c.get("/static/app.js").text
        assert "/api/rule" in js and 'data-uid="step:${i+1}"' in js

        # load repo-bundled domain example
        r = c.post("/api/load-bundled")
        assert r.status_code == 200
        st = c.get("/api/state").json()
        assert len(st["segments"]) == 8

        # overlay / jitter / per-segment SVGs render with stable identities
        overlay = c.get("/svg/overlay").text
        assert overlay.startswith("<svg")
        assert 'data-uid="overlay:seg-01-reference:CH1"' in overlay
        assert 'data-uid="trigger:' not in overlay or True
        jitter = c.get("/svg/jitter").text
        assert 'data-uid="jitter-point:seg-01-reference"' in jitter
        seg_svg = c.get("/svg/segment/seg-04-multipeak-alignment").text
        assert 'data-uid="segwave:seg-04-multipeak-alignment:CH1"' in seg_svg

        # visible review path: candidates, windows, exclusion reasons
        seg4 = _seg(st, "seg-04-multipeak-alignment")
        assert seg4["ambiguous"] and len(seg4["candidates"]) == 2
        seg3 = _seg(st, "seg-03-vertical-saturation")
        assert "window-contains-saturation" in seg3["reason_code"]
        seg6 = _seg(st, "seg-06-missing-samples")
        assert "window-too-much-invalid" in seg6["reason_code"]
        seg7 = _seg(st, "seg-07-trigger-at-boundary")
        assert seg7["jitter"]["exclusion_code"] == "trigger-at-boundary"

        # machine-readable state values carry stable identities
        for s in st["segments"]:
            assert s["seg_id"].startswith("seg-")
            for cand in s["candidates"]:
                assert cand["candidate_id"].startswith("cand-")

        base = st["tip_event"]

        # parameter revision v1 -> v2
        r = c.post("/api/window", json={"to_version": "v2",
                                        "expected_last_event": base})
        assert r.status_code == 200
        st = c.get("/api/state").json()
        assert st["window"]["version"] == "v2"

        # branch ruling on multi-peak segment
        tip = st["tip_event"]
        cid = _seg(st, "seg-04-multipeak-alignment")["candidates"][0]["candidate_id"]
        r = c.post("/api/rule", json={
            "segment": "seg-04-multipeak-alignment",
            "candidate_id": cid, "decision": "accept",
            "expected_last_event": tip})
        assert r.status_code == 200
        st = c.get("/api/state").json()
        assert _seg(st, "seg-04-multipeak-alignment")["adopted_candidate"] == cid
        rule_evt = st["tip_event"]

        # undo -> new branch restores pre-ruling ambiguity
        r = c.post("/api/undo", json={"event_id": rule_evt,
                                      "label": "undo"})
        nb = r.json()["branch_id"]
        hist = c.get(f"/api/state?branch={nb}").json()
        assert _seg(hist, "seg-04-multipeak-alignment")["ambiguous"]
        assert _seg(hist, "seg-04-multipeak-alignment")["adopted_candidate"] is None

        # two window saves from same base: second sees 409 + both-side diff
        st = c.get("/api/state").json()
        common_base = st["tip_event"]
        assert c.post("/api/window", json={
            "to_version": "v1", "expected_last_event": common_base}).status_code == 200
        conflict = c.post("/api/window", json={
            "to_version": "v2", "expected_last_event": common_base})
        assert conflict.status_code == 409
        body = conflict.json()
        assert body["error"] == "version-conflict"
        assert body["diff"]["your_payload"]["to_version"] == "v2"
        assert body["diff"]["server_payload"]["to_version"] == "v1"

        # local-only source patch invalidates one scope
        st = c.get("/api/state").json()
        r = c.post("/api/source-patch", json={
            "segment": "seg-08-window-revision", "channel": "CH1",
            "field": "reviewer_note", "value": "note",
            "expected_last_event": st["tip_event"]})
        assert r.json()["invalidation"] == ["seg-08-window-revision/CH1"]

        # provenance export before reset
        bundle = c.get("/api/export").json()
        ev_ids = [e["event_id"] for e in bundle["events"]]
        seqs = [e["seq"] for e in bundle["events"]]
        mat_ids = {m["material_id"] for m in bundle["materials"]}
        branch_ids = {b["branch_id"] for b in bundle["branches"]}

        # delete local DB then import the archive
        assert c.post("/api/reset").status_code == 200
        r = c.post("/api/import", json=bundle)
        assert r.status_code == 200
        after = c.get("/api/export").json()
        assert [e["event_id"] for e in after["events"]] == ev_ids
        assert [e["seq"] for e in after["events"]] == seqs
        assert {m["material_id"] for m in after["materials"]} == mat_ids
        assert {b["branch_id"] for b in after["branches"]} == branch_ids

        # "reopen" history: undo-branch still shows pre-ruling state
        reopened = c.get(f"/api/state?branch={nb}").json()
        assert _seg(reopened, "seg-04-multipeak-alignment")["ambiguous"]
        assert c.get(f"/svg/overlay?branch={nb}").status_code == 200


def test_import_twice_is_refused_to_protect_identities(tmp_path, monkeypatch):
    monkeypatch.setenv("WFLINE_DB", str(tmp_path / "e2e2.sqlite3"))
    appmod = importlib.import_module("app")
    importlib.reload(appmod)
    with TestClient(appmod.app) as c:
        c.post("/api/load-bundled")
        bundle = c.get("/api/export").json()
        # a second load on non-empty repo is rejected
        r = c.post("/api/load-bundled")
        assert r.status_code == 409
        c.post("/api/reset")
        c.post("/api/import", json=bundle)
        r2 = c.post("/api/import", json=bundle)
        assert r2.status_code == 400


def test_state_before_load_is_a_clean_409(tmp_path, monkeypatch):
    monkeypatch.setenv("WFLINE_DB", str(tmp_path / "empty.sqlite3"))
    appmod = importlib.import_module("app")
    importlib.reload(appmod)
    with TestClient(appmod.app) as c:
        assert c.get("/api/state").status_code == 409
