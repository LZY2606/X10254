"""Branch replay: material assembly, rulings, revisions and time travel."""
from __future__ import annotations

import copy
import json
from typing import Any

from .domain.derive import derive_snapshot
from .store import Store, canonical_json, content_hash


def _material_from_nodes(nodes: list[dict]) -> dict:
    """Assemble current material on a branch; old node bytes remain readable."""
    by_address: dict[str, dict] = {}
    for node in sorted(nodes, key=lambda n: (n["ordinal"], n["node_id"])):
        if node["kind"] in {"channel", "segment", "window"}:
            bucket = {
                "channel": "channels",
                "segment": "segments",
                "window": "windows",
            }[node["kind"]]
            by_address.setdefault(bucket, {})[node["address"]] = node["content"]
        else:
            by_address[node["address"]] = node["content"]
    material: dict[str, Any] = {}
    material.update(by_address.get("root", {}))
    material["channels"] = list(by_address.get("channels", {}).values())
    material["segments"] = list(by_address.get("segments", {}).values())
    material["windows"] = list(by_address.get("windows", {}).values())
    return material


def _apply_event(material: dict, revisions: dict[str, dict],
                 interpretation: dict, event: dict, nodes_by_id: dict[str, dict]) -> None:
    etype = event["type"]
    payload = event["payload"]
    if etype == "parameter_revision":
        revisions[payload["address"]] = {
            "value": payload["value"], "event_id": event["event_id"],
            "from_value": payload.get("from_value"),
            "basis": payload.get("basis"),
        }
    elif etype == "candidate_ruling":
        interpretation[payload["candidate_id"]] = payload["decision"]
        interpretation[payload["candidate_id"] + ":event"] = event["event_id"]
    elif etype == "window_revision":
        win = next(w for w in material["windows"]
                   if w["id"] == payload["window_id"]
                   and int(w.get("version", 1)) == int(payload["from_version"]))
        new_version = int(payload["from_version"]) + 1
        material["windows"].append({
            **win, "version": new_version,
            "start": payload["start"], "end": payload["end"],
            "start_s": payload["start_s"], "end_s": payload["end_s"],
            "revised_by": event["event_id"],
        })
    elif etype == "source_amend":
        node = nodes_by_id[payload["node_id"]]
        content = node["content"]
        if node["kind"] == "segment":
            material["segments"] = [
                content if s["id"] == content["id"] else s
                for s in material["segments"]
            ]
        elif node["kind"] == "channel":
            material["channels"] = [
                content if c["id"] == content["id"] else c
                for c in material["channels"]
            ]
        elif node["kind"] == "window":
            key = (content["id"], int(content.get("version", 1)))
            material["windows"] = [
                w for w in material["windows"]
                if (w["id"], int(w.get("version", 1))) != key
            ]
            material["windows"].append(content)
    elif etype == "branch_note":
        pass
    else:
        raise ValueError(f"unknown event type {etype}")
