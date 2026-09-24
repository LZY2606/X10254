"""Counterexample archive: fixed case replay, provenance and invalidation."""
import json

import pytest

from appcore.casegen import build_case
from appcore.ingest import ingest_case, validate_case
from appcore.projection import invalidation_for, project
from appcore.repository import Conflict, Repository
from appcore import services as svc
from appcore import waveforms as W

EXPECTED_CODES = {
    "time-uneven", "saturation-skipped", "multipeak-candidates",
    "channel-deskew-applied", "gap-skipped", "trigger-at-boundary",
    "bad-probe-attenuation",
}


@pytest.fixture()
def loaded():
    case = build_case()
    repo = Repository(":memory:")
    ingest_case(repo, case)
    return repo, case


def _state(repo, case, branch="main"):
    return project(repo.branch_events(branch), case)


def _seg(state, sid):
    return next(s for s in state["segments"] if s["seg_id"] == sid)


def test_fixed_case_passes_all_seven_situations(loaded):
    repo, case = loaded
    state = _state(repo, case)
    codes = {d["code"] for s in state["segments"] for d in s["diagnostics"]}
    missing = EXPECTED_CODES - codes
    assert not missing, f"counterexample archive misses {missing}"


def test_uneven_segment_alignment_is_time_aware(loaded):
    repo, case = loaded
    state = _state(repo, case)
    seg = _seg(state, "seg-02-uneven-timebase")
    lag = seg["candidates"][0]["lag_ns"]
    # true event center 60.4 ns on explicit axis; naive index would bias ~2 ns
    assert abs(lag - 60.4) < 0.3


def test_saturation_segment_excluded_with_reason_kept_for_review(loaded):
    repo, case = loaded
    state = _state(repo, case)
    seg = _seg(state, "seg-03-vertical-saturation")
    assert seg["included"] is False
    assert "window-contains-saturation" in seg["reason_code"]
    assert "voltage-unconvertible" in seg["reason_code"]
    assert seg["jitter"]["included"] is False
    # original rail samples are still in the case (not dropped for plotting)
    raw = case["segments"][2]["channels"]["CH1"]["samples"]
    assert W.SAT_HIGH in raw


def test_multipeak_adopted_and_not_adopted_both_retained(loaded):
    repo, case = loaded
    state = _state(repo, case)
    seg4 = _seg(state, "seg-04-multipeak-alignment")
    cids = [c["candidate_id"] for c in seg4["candidates"]]
    assert len(cids) == 2
    svc.rule_candidate(repo, segment="seg-04-multipeak-alignment",
                       candidate_id=cids[0], decision="accept",
                       note="reviewer picks near-trigger peak")
    state2 = _state(repo, case)
    seg4b = _seg(state2, "seg-04-multipeak-alignment")
    adopted = [c for c in seg4b["candidates"] if c["adopted"]]
    notadopted = [c for c in seg4b["candidates"] if not c["adopted"]]
    assert len(adopted) == 1 and len(notadopted) == 1
    decision = [d for d in seg4b["diagnostics"]
                if d["kind"] == "decision"]
    assert decision and decision[0]["code"] == "candidate-adopted"


def test_undo_fork_restores_pre_ruling_branch(loaded):
    repo, case = loaded
    state = _state(repo, case)
    cid = _seg(state, "seg-04-multipeak-alignment")["candidates"][0]["candidate_id"]
    svc.rule_candidate(repo, segment="seg-04-multipeak-alignment",
                       candidate_id=cid, decision="accept")
    rule_evt = repo.branch_events("main")[-1]["event_id"]
    forked = svc.undo_to_before_event(repo, event_id=rule_evt,
                                      label="undo-ruling")
    hist = _state(repo, case, forked["branch_id"])
    assert _seg(hist, "seg-04-multipeak-alignment")["adopted_candidate"] is None
    assert _seg(hist, "seg-04-multipeak-alignment")["ambiguous"] is True
    # main still carries the ruling (history not overwritten)
    main = _state(repo, case, "main")
    assert _seg(main, "seg-04-multipeak-alignment")["adopted_candidate"] == cid


def test_trigger_at_boundary_excluded_from_jitter_kept_in_overlay(loaded):
    repo, case = loaded
    state = _state(repo, case)
    seg = _seg(state, "seg-07-trigger-at-boundary")
    assert seg["included"] is True
    assert seg["jitter"]["included"] is False
    assert seg["jitter"]["exclusion_code"] == "trigger-at-boundary"


def test_gap_segment_excluded_no_interpolation(loaded):
    repo, case = loaded
    state = _state(repo, case)
    seg = _seg(state, "seg-06-missing-samples")
    assert seg["included"] is False
    assert "window-too-much-invalid" in seg["reason_code"]


def test_window_revision_changes_measurement_not_history(loaded):
    repo, case = loaded
    seg8 = "seg-08-window-revision"
    before = _seg(_state(repo, case), seg8)
    v1 = before["measurements"]["CH1"]["vpp"]
    svc.revise_window(repo, to_version="v2")
    after = _seg(_state(repo, case), seg8)
    assert after["measurements"]["CH1"]["window_version"] == "v2"
    assert after["measurements"]["CH1"]["vpp"] < v1  # negative bump outside v2
    # historical event (ingest) still projects to v1
    ingest_id = repo.branch_events("main")[0]["event_id"]
    hb = svc.fork_branch(repo, base_branch="main", label="h",
                         at_event=ingest_id)["branch_id"]
    hist = _seg(_state(repo, case, hb), seg8)
    assert hist["measurements"]["CH1"]["window_version"] == "v1"
    assert hist["measurements"]["CH1"]["vpp"] == v1


def test_source_patch_invalidation_is_local_only(loaded):
    repo, case = loaded
    scope = invalidation_for("source-patched",
                             {"segment": "seg-08-window-revision",
                              "channel": "CH1"})
    assert scope == ["seg-08-window-revision/CH1"]
    assert all("seg-01" not in k for k in scope)
    svc.patch_source(repo, segment="seg-08-window-revision", channel="CH1",
                     field="reviewer_note",
                     change={"reviewer_note": "annotated"})
    state = _state(repo, case)
    evt = state["event_log"][-1]
    assert evt["invalidation"] == ["seg-08-window-revision/CH1"]
    seg8 = _seg(state, "seg-08-window-revision")
    assert seg8["source_materials"]["CH1"]["patched"] is True
    seg1 = _seg(state, "seg-01-reference")
    assert seg1["source_materials"]["CH1"]["patched"] is False


def test_optimistic_conflict_two_window_saves(loaded):
    repo, case = loaded
    base = repo.branch_events("main")[-1]["event_id"]
    svc.revise_window(repo, to_version="v1", expected_last_event=base)
    with pytest.raises(Conflict) as ei:
        svc.revise_window(repo, to_version="v2",
                          expected_last_event=base)
    assert "server_tip" in ei.value.diff
    assert ei.value.diff["your_payload"]["to_version"] == "v2"


def test_reinterpretation_keeps_old_branch_and_raw_materials(loaded):
    repo, case = loaded
    mats_before = {m["material_id"] for m in repo.list_materials()}
    svc.patch_source(repo, segment="seg-01-reference", channel="CH1",
                     field="reviewer_note", change={"reviewer_note": "x"})
    mats_after = {m["material_id"] for m in repo.list_materials()}
    assert mats_before <= mats_after
    assert len(repo.list_branches()) >= 1


def test_ingest_is_atomic_and_rolls_back():
    case = build_case()
    case["segments"][0]["channels"]["CH1"].pop("timebase")
    assert validate_case(case)
    repo = Repository(":memory:")
    with pytest.raises(ValueError):
        ingest_case(repo, case)
    assert repo.all_events() == []
    assert repo.list_materials() == []


def test_export_delete_import_preserves_identity_order_provenance(tmp_path):
    case = build_case()
    db = tmp_path / "a.sqlite3"
    repo = Repository(str(db))
    ingest_case(repo, case)
    state = _state(repo, case)
    cid = _seg(state, "seg-04-multipeak-alignment")["candidates"][0]["candidate_id"]
    svc.rule_candidate(repo, segment="seg-04-multipeak-alignment",
                       candidate_id=cid, decision="accept")
    bundle = repo.export_bundle()
    repo.close()

    db2 = tmp_path / "b.sqlite3"
    repo2 = Repository(str(db2))
    n = repo2.import_bundle(bundle)
    assert n["events"] >= 2
    after = repo2.export_bundle()
    assert [e["event_id"] for e in after["events"]] == \
           [e["event_id"] for e in bundle["events"]]
    assert [e["seq"] for e in after["events"]] == \
           [e["seq"] for e in bundle["events"]]
    assert {m["material_id"] for m in after["materials"]} == \
           {m["material_id"] for m in bundle["materials"]}
    state2 = _state(repo2, case)
    assert _seg(state2, "seg-04-multipeak-alignment")["adopted_candidate"] == cid


def test_machine_readable_judgement_events(loaded):
    repo, case = loaded
    state = _state(repo, case)
    diags = [d for s in state["segments"] for d in s["diagnostics"]]
    for d in diags:
        if d["kind"] == "judgement":
            assert d["rule"] == W.RULE_VERSION
            assert isinstance(d["code"], str) and d["code"]


def test_every_domain_value_has_unit_or_basis(loaded):
    repo, case = loaded
    assert "time" in case["coordinate_basis"]
    for seg in case["segments"]:
        for cid in ("CH1", "CH2"):
            ch = seg["channels"][cid]
            assert ch["timebase"].get("unit")
            assert ch["vertical_gain"] and ch["probe_attenuation"]
            assert ch["deskew"]
        assert seg["channels"]["trigger"]["timestamp"].endswith("ns")
    for w in case["windows"]:
        assert w["start"].endswith("ns") and w["end"].endswith("ns")


def test_reverse_trace_material_rule_decision(loaded):
    """From final state trace back to material id, rule version, decision."""
    repo, case = loaded
    state = _state(repo, case)
    cid = _seg(state, "seg-04-multipeak-alignment")["candidates"][0]["candidate_id"]
    svc.rule_candidate(repo, segment="seg-04-multipeak-alignment",
                       candidate_id=cid, decision="accept",
                       note="human call")
    state2 = _state(repo, case)
    seg = _seg(state2, "seg-04-multipeak-alignment")
    dec = next(d for d in seg["diagnostics"]
               if d["kind"] == "decision" and d["code"] == "candidate-adopted")
    ruling_evt = repo.get_event(dec["event_id"])
    assert ruling_evt["type"] == "candidate-ruling"
    ingest = next(e for e in repo.branch_events("main")
                  if e["type"] == "ingest")
    materials = json.loads(ingest["payload"])["materials"]
    slot = f"seg-04-multipeak-alignment/CH1/samples"
    assert slot in materials
    assert dec["rule"] == W.RULE_VERSION
