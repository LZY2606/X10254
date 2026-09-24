"""Application services: rulings, undo/branch, window revisions, patches."""
from __future__ import annotations

import json

from appcore.repository import Conflict, Repository, stable_id
from appcore.projection import invalidation_for


def branch_tip(repo: Repository, branch_id: str) -> str | None:
    events = repo.branch_events(branch_id)
    return events[-1]["event_id"] if events else None


def revise_window(repo: Repository, *, to_version: str,
                  branch_id: str = "main",
                  expected_last_event: str | None = None) -> dict:
    payload = {"to_version": to_version,
               "scope": "measurement-window",
               "invalidation": invalidation_for("window-revised", {})}
    # Conflict diff is built by the repository (server_tip + both payloads),
    # which keeps both reviewers' versions visible in the 409 response.
    return repo.append(branch_id, "window-revised", payload,
                       expected_last_event=expected_last_event)


def patch_source(repo: Repository, *, segment: str, channel: str,
                 field: str, change: dict, branch_id: str = "main",
                 expected_last_event: str | None = None) -> dict:
    payload = {"segment": segment, "channel": channel, "field": field,
               "change": change,
               "invalidation": invalidation_for(
                   "source-patched",
                   {"segment": segment, "channel": channel})}
    return repo.append(branch_id, "source-patched", payload,
                       expected_last_event=expected_last_event)


def rule_candidate(repo: Repository, *, segment: str, candidate_id: str,
                   decision: str, note: str = "", branch_id: str = "main",
                   expected_last_event: str | None = None) -> dict:
    payload = {"segment": segment, "candidate_id": candidate_id,
               "decision": decision, "adopted": decision == "accept",
               "note": note,
               "invalidation": invalidation_for(
                   "candidate-ruling", {"segment": segment})}
    return repo.append(branch_id, "candidate-ruling", payload,
                       expected_last_event=expected_last_event,
                       conflict_diff={"scope": "candidate-ruling",
                                      "segment": segment})


def undo_to_before_event(repo: Repository, *, event_id: str,
                         branch_id: str = "main", label: str = "undo") -> dict:
    """Fork the branch at the event *before* ``event_id`` (undo support).

    The old branch, ruling and referenced raw bytes stay readable; the new
    branch replays history truncated to the fork point.
    """
    events = repo.branch_events(branch_id)
    ids = [e["event_id"] for e in events]
    if event_id not in ids:
        raise ValueError("event not visible on branch")
    pos = ids.index(event_id)
    fork_event = events[pos - 1]["event_id"] if pos > 0 else None
    new_branch = repo.create_branch(branch_id, fork_event, label)
    return {"branch_id": new_branch, "forked_from_event": fork_event,
            "based_on": branch_id}


def fork_branch(repo: Repository, *, base_branch: str, label: str,
                at_event: str | None = None) -> dict:
    if at_event is None:
        at_event = branch_tip(repo, base_branch)
    return {"branch_id": repo.create_branch(base_branch, at_event, label),
            "forked_from_event": at_event, "based_on": base_branch}
