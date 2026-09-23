"""Unit tests: derivation rules, atomic import, undo, conflict, identity."""
import json

import pytest

import core
import db
import derive
from app import create_app, import_case, load_case_file


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


@pytest.fixture()
def loaded(conn):
    imp_id, bid = import_case(conn, load_case_file())
    conn.commit()
    core.refresh_derived(conn, bid)
    return bid


def state_and_derived(conn, bid):
    state, chain, meta = core.replay(conn, bid)
    return state, derive.compute_derived(state, meta)


# R1: non-uniform timebase must not use index subtraction
def test_nonuniform_timebase_uses_actual_timestamps():
    t = [0.0, 0.2, 0.45, 0.61, 0.9]  # non-uniform
    deltas = derive.time_deltas(t)
    assert deltas == pytest.approx([0.2, 0.25, 0.16, 0.29])
    assert len(set(round(d, 6) for d in deltas)) > 1  # not index * const


def test_nonuniform_segment_alignment_recovers_lag(loaded, conn):
    state, derived = state_and_derived(conn, loaded)
    seg2 = next(s for s in state["segments"].values() if s["key"] == "seg-002")
    t = seg2["raw"]["t"]["values"]
    assert len(set(round(d, 4) for d in derive.time_deltas(t))) > 1
    al = derived["alignment"][seg2["id"]]
    assert al["candidates"], "non-uniform segment must still align via timestamps"
    assert abs(al["candidates"][0]["lag_ns"]) < 0.3


# R2: missing/saturated intervals never enter interpolation
def test_interp_excludes_missing_and_saturated():
    t = [float(i) for i in range(10)]
    v = [float(i * 2) for i in range(10)]
    out = derive.interp_excluding(t, v, excluded=[4, 5], tq=[2.5, 4.0, 4.5, 5.5, 6.5])
    assert out[0] == pytest.approx(5.0)   # normal interval interpolates
    assert out[1] is None                 # excluded sample itself
    assert out[2] is None                 # bracketed by excluded samples
    assert out[3] is None                 # bracket touches excluded sample
    assert out[4] == pytest.approx(13.0)  # clear of the gap


def test_saturated_segment_flagged_in_case(loaded, conn):
    state, _ = state_and_derived(conn, loaded)
    seg3 = next(s for s in state["segments"].values() if s["key"] == "seg-003")
    assert seg3["raw"]["flags"]["saturated"], "case must contain vertical saturation"


# R3: multi-peak alignment keeps all legitimate candidates
def test_multipeak_alignment_candidates_kept(loaded, conn):
    state, derived = state_and_derived(conn, loaded)
    seg7 = next(s for s in state["segments"].values() if s["key"] == "seg-007")
    cands = derived["alignment"][seg7["id"]]["candidates"]
    assert len(cands) >= 2, "double-pulse segment must yield multiple peaks"
    assert all(c["status"] == "pending" for c in cands)
    lags = sorted(c["lag_ns"] for c in cands)
    assert lags[-1] - lags[0] == pytest.approx(1.2, abs=0.3)


def test_deskew_applied_in_derived_layer_only(loaded, conn):
    state, derived = state_and_derived(conn, loaded)
    seg4 = next(s for s in state["segments"].values() if s["key"] == "seg-004")
    assert seg4["raw"]["deskew"] == {"value": 0.3, "unit": "ns"}  # raw untouched
    norm = derived["normalized"][seg4["id"]]
    assert norm["t"][0] == pytest.approx(-0.3)


def test_boundary_trigger_depends_on_window_version():
    case = load_case_file()
    v1 = case["meas_windows"]["mw-v1"]
    v2 = case["meas_windows"]["mw-v2"]
    ok1, reason1 = derive.window_check(10.0, v1)
    ok2, _ = derive.window_check(10.0, v2)
    assert not ok1 and "boundary" in reason1
    assert ok2


def test_unconvertible_unit_stays_raw_with_diagnostic(loaded, conn):
    state, derived = state_and_derived(conn, loaded)
    seg8 = next(s for s in state["segments"].values() if s["key"] == "seg-008")
    assert seg8["raw"]["v"]["unit"] == "div"  # raw preserved in place
    assert seg8["id"] not in derived["normalized"]
    reasons = [e for e in derived["exclusions"] if e["segment"] == "seg-008"]
    assert reasons and reasons[0]["rule"] == "unit.unconvertible"
    assert any(d["segment"] == "seg-008" for d in derived["diagnostics"])


def test_import_rollback_is_atomic(conn):
    case = load_case_file()
    case["segments"][3].pop("trigger_ts")  # corrupt one segment
    with pytest.raises(ValueError):
        import_case(conn, case)
    assert conn.execute("SELECT COUNT(*) c FROM segments").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM imports").fetchone()["c"] == 0


def test_undo_restores_prior_adjudication(loaded, conn):
    eid = db.append_event(conn, loaded, "adjudicate",
                          {"candidate_id": "cand:seg-007#1", "decision": "accept"})
    conn.commit()
    state, _ = state_and_derived(conn, loaded)
    assert state["adjudications"]["cand:seg-007#1"]["decision"] == "accept"
    db.append_event(conn, loaded, "undo", {"target": eid})
    conn.commit()
    state, _ = state_and_derived(conn, loaded)
    assert "cand:seg-007#1" not in state["adjudications"]


def test_conflict_error_carries_both_sides(loaded, conn):
    head = conn.execute("SELECT head_event_id h FROM branches WHERE id=?",
                        (loaded,)).fetchone()["h"]
    db.append_event(conn, loaded, "adjudicate",
                    {"candidate_id": "cand:seg-007#1", "decision": "accept"},
                    expected_head=head)
    conn.commit()
    with pytest.raises(db.ConflictError) as ei:
        db.append_event(conn, loaded, "adjudicate",
                        {"candidate_id": "cand:seg-007#1", "decision": "reject"},
                        expected_head=head)
    assert ei.value.their_events, "later submitter must see the other side"


def test_export_import_preserves_identity(loaded, conn):
    db.append_event(conn, loaded, "adjudicate",
                    {"candidate_id": "cand:seg-007#2", "decision": "reject"})
    conn.commit()
    core.refresh_derived(conn, loaded)
    before_segments = [dict(r) for r in conn.execute(
        "SELECT * FROM segments ORDER BY ordinal").fetchall()]
    before_events = [dict(r) for r in conn.execute(
        "SELECT * FROM events ORDER BY seq").fetchall()]
    dump = {"format": "waveform-overlay-export/1", "tables": {}}
    for t in db.TABLES:
        dump["tables"][t] = [dict(r) for r in conn.execute(
            "SELECT * FROM %s" % t).fetchall()]
    for t in db.TABLES:
        conn.execute("DELETE FROM %s" % t)
    conn.commit()
    for t in db.TABLES:
        for row in dump["tables"][t]:
            cols = list(row.keys())
            conn.execute("INSERT INTO %s(%s) VALUES(%s)"
                         % (t, ",".join(cols), ",".join("?" * len(cols))),
                         [row[c] for c in cols])
    conn.commit()
    after_segments = [dict(r) for r in conn.execute(
        "SELECT * FROM segments ORDER BY ordinal").fetchall()]
    after_events = [dict(r) for r in conn.execute(
        "SELECT * FROM events ORDER BY seq").fetchall()]
    assert before_segments == after_segments
    assert before_events == after_events
    state, derived = state_and_derived(conn, loaded)
    assert state["adjudications"]["cand:seg-007#2"]["decision"] == "reject"


def test_window_revision_invalidates_only_window_scope(loaded, conn):
    db.append_event(conn, loaded, "window_revision",
                    {"from": "mw-v1", "to": "mw-v2"})
    conn.commit()
    core.refresh_derived(conn, loaded)
    scope = core.invalidation_scope(conn, loaded)
    kinds = {item["kind"] for item in scope}
    assert kinds <= {"window", "exclusion"}, scope
    assert any(i["id"] == "win:seg-006@mw-v1" for i in scope)
    state, derived = state_and_derived(conn, loaded)
    seg6 = next(s for s in state["segments"].values() if s["key"] == "seg-006")
    assert derived["windows"][seg6["id"]]["included"] is True


def test_source_patch_invalidates_only_local_dependency(loaded, conn):
    state, _ = state_and_derived(conn, loaded)
    seg4 = next(s for s in state["segments"].values() if s["key"] == "seg-004")
    db.append_event(conn, loaded, "source_patch",
                    {"segment_id": seg4["id"],
                     "patch": {"deskew": {"value": 0.25, "unit": "ns"}}})
    conn.commit()
    core.refresh_derived(conn, loaded)
    scope = core.invalidation_scope(conn, loaded)
    subjects = {item["subject"] for item in scope}
    assert subjects == {"seg-004"}, scope
