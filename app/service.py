"""Application operations over the event log: branches, decisions and scoped
source revisions. Every mutation is one appended event; undo is another
event (reset_decision) that restores the previous branch state."""
from __future__ import annotations

import json
import uuid

from . import ingest, state as st, storage


def new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex[:12]}"


def ensure_main(conn) -> bool:
    """Create the main branch and import the bundled case if the DB is empty."""
    if storage.get_branch(conn, st.MAIN_BRANCH) is not None:
        return False
    storage.create_branch(conn, st.MAIN_BRANCH, "main", None, 0)
    case = ingest.load_case_dir()
    event_id = new_id("evt")
    ingest.import_case(conn, st.MAIN_BRANCH, case, event_id, 1)
    return True


def fork_branch(conn, parent_branch_id: str, name: str) -> dict:
    parent = storage.get_branch(conn, parent_branch_id)
    if parent is None:
        raise KeyError(parent_branch_id)
    branch_id = new_id("br")
    head = parent["head_seq"]
    storage.create_branch(conn, branch_id, name, parent_branch_id, head)
    storage.append_event(conn, branch_id, new_id("evt"), "branch_fork", {
        "new_branch_id": branch_id, "name": name,
        "fork_seq": head, "parent_branch_id": parent_branch_id})
    return storage.get_branch(conn, branch_id)


def _revision(conn, branch_id, event_type, fact_type, fact_key, payload,
              reason, expected_head, blob_bytes=None):
    """Pre-compute the invalidation scope from the *pre-revision* state, write
    the new fact version, then append the revision event."""
    before = st.state_at(conn, branch_id)
    if before["fact_versions"].get(fact_key) is None:
        raise KeyError(f"unknown fact {fact_key}")
    seq_before = before["upto_seq"]
    event_id = new_id("evt")
    seq = seq_before + 1
    fid = ingest.revise_fact(conn, branch_id, event_type, fact_type, fact_key,
                             payload, seq, event_id, reason,
                             preserve_blob=blob_bytes is not None,
                             blob_bytes=blob_bytes)
    impact = st.impact_of(before, fact_key)
    storage.append_event(conn, branch_id, event_id, event_type, {
        "fact_key": fact_key, "new_fact_id": fid, "reason": reason,
        "invalidation_scope": impact, "n_affected": len(impact),
        "rule_version": before["rule_version"],
    }, expected_head=expected_head)
    after = st.state_at(conn, branch_id)
    return {"event_id": event_id, "fact_id": fid,
            "invalidation_scope": impact, "state_hash": after["state_hash"]}


def revise_source_segment(conn, branch_id, segment_id, new_segment, reason,
                          expected_head=None):
    raw_bytes = json.dumps(new_segment, sort_keys=True,
                           ensure_ascii=False).encode("utf-8")
    return _revision(conn, branch_id, "revise_source", "segment",
                     f"seg:{segment_id}", new_segment, reason, expected_head,
                     blob_bytes=raw_bytes)


def revise_calibration(conn, branch_id, channel_id, changes, reason,
                       expected_head=None):
    before = st.state_at(conn, branch_id)
    fact_id = before["fact_versions"].get(f"calib:{channel_id}")
    if fact_id is None:
        raise KeyError(channel_id)
    old = json.loads(next(
        f for f in storage.facts(conn) if f["fact_id"] == fact_id)["payload_json"])
    payload = {**old, **changes, "version_note": reason}
    return _revision(conn, branch_id, "revise_calibration", "calibration",
                     f"calib:{channel_id}", payload, reason, expected_head)


def revise_window(conn, branch_id, window_id, changes, reason,
                  expected_head=None):
    before = st.state_at(conn, branch_id)
    fact_id = before["fact_versions"].get(f"win:{window_id}")
    if fact_id is None:
        raise KeyError(window_id)
    old = json.loads(next(
        f for f in storage.facts(conn) if f["fact_id"] == fact_id)["payload_json"])
    payload = {**old, **changes, "version_note": reason}
    return _revision(conn, branch_id, "revise_window", "window",
                     f"win:{window_id}", payload, reason, expected_head)


def revise_params(conn, branch_id, changes, reason, expected_head=None):
    before = st.state_at(conn, branch_id)
    fact_id = before["fact_versions"]["params:alignment"]
    old = json.loads(next(
        f for f in storage.facts(conn) if f["fact_id"] == fact_id)["payload_json"])
    payload = {**old, **changes, "version_note": reason}
    return _revision(conn, branch_id, "revise_params", "params",
                     "params:alignment", payload, reason, expected_head)


def decide_candidate(conn, branch_id, candidate_id, status, note=None,
                     expected_head=None):
    if status not in ("accepted", "rejected"):
        raise ValueError(status)
    current = st.state_at(conn, branch_id)
    known = {c["candidate_id"]
             for cl in current["candidates"].values() for c in cl}
    if candidate_id not in known:
        raise KeyError(candidate_id)
    event_id = new_id("evt")
    storage.append_event(conn, branch_id, event_id, "decide", {
        "candidate_id": candidate_id, "status": status, "note": note,
    }, expected_head=expected_head)
    return event_id


def reset_decision(conn, branch_id, candidate_id, expected_head=None):
    event_id = new_id("evt")
    storage.append_event(conn, branch_id, event_id, "reset_decision", {
        "candidate_id": candidate_id,
    }, expected_head=expected_head)
    return event_id
