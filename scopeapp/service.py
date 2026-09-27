"""High-level operations: import, append events, branch, time travel, export."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from .replay import _apply_event, _material_from_nodes
from .domain.derive import derive_snapshot
from .store import Store, canonical_json, content_hash


class ConflictError(Exception):
    def __init__(self, message: str, expected_seq: int, actual_seq: int, diff: dict):
        super().__init__(message)
        self.expected_seq = expected_seq
        self.actual_seq = actual_seq
        self.diff = diff


class NotFoundError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ScopeService:
    def __init__(self, store: Store):
        self.store = store

    # ----- import --------------------------------------------------------
    def import_case(self, case: dict, branch_id: str = "main",
                    title: str = "main review line") -> int:
        """Transactional import. Any failure rolls back the whole batch."""
        nodes = self._nodes_from_case(case)
        with self.store.lock():
            if self.store.branch_exists(branch_id):
                raise ValueError(f"branch {branch_id} already exists")
            seq = 0
            with self.store.connect() as conn:
                try:
                    self._write_nodes(conn, branch_id, nodes, 0)
                    conn.execute(
                        "INSERT INTO branches(branch_id, material_id, parent_branch,"
                        " parent_seq, title, created_seq, ordinal)"
                        " VALUES(?,?,?,?,?,?,?)",
                        (branch_id, case["id"], None, None, title, 0,
                         self._next_branch_ordinal(conn)),
                    )
                    material = _material_from_nodes(self._read_nodes(conn, branch_id))
                    self._validate_case(case)
                    snap = derive_snapshot(material, {}, {})
                    self._write_snapshot(conn, branch_id, 0, snap)
                except Exception:
                    raise
            return seq

    def _validate_case(self, case: dict) -> None:
        required = {"id", "acquisition", "channels", "segments", "windows"}
        missing = required - set(case)
        if missing:
            raise ValueError(f"case missing required sections: {sorted(missing)}")
        for segment in case["segments"]:
            if "samples" not in segment or "trigger_offset" not in segment:
                raise ValueError(f"segment {segment.get('id')} incomplete")

    def _nodes_from_case(self, case: dict) -> list[dict]:
        nodes: list[dict] = []
        root = {k: v for k, v in case.items()
                if k not in {"channels", "segments", "windows"}}
        nodes.append({"kind": "root", "address": "root", "content": root})
        for channel in case["channels"]:
            nodes.append({"kind": "channel", "address": channel["id"],
                          "content": channel})
        for segment in case["segments"]:
            nodes.append({"kind": "segment", "address": segment["id"],
                          "content": segment})
        for window in case["windows"]:
            address = f"{window['id']}:v{window.get('version', 1)}"
            nodes.append({"kind": "window", "address": address,
                          "content": window})
        for ordinal, node in enumerate(nodes):
            node["ordinal"] = ordinal
            node["node_id"] = self._node_id(node)
        return nodes

    @staticmethod
    def _node_id(node: dict) -> str:
        return content_hash([node["kind"], node["address"], node["content"]])

    def _write_nodes(self, conn, branch_id: str, nodes: list[dict],
                    start_ordinal: int, supersedes: str | None = None) -> None:
        for node in nodes:
            conn.execute(
                "INSERT INTO source_nodes(node_id, branch_id, kind, address,"
                " content_hash, bytes_json, ordinal, supersedes)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (node["node_id"], branch_id, node["kind"], node["address"],
                 content_hash(node["content"]), canonical_json(node["content"]),
                         start_ordinal + int(node["ordinal"]), supersedes),
            )

    # ----- replay --------------------------------------------------------
    def replay(self, branch_id: str, seq: int | None = None) -> dict:
        """Replay events up to ``seq`` and return the historical snapshot."""
        with self.store.connect() as conn:
            nodes = self.store._read_nodes(conn, branch_id)
            rows = conn.execute(
                "SELECT * FROM events WHERE branch_id=? ORDER BY seq",
                (branch_id,),
            ).fetchall()
        events = [{"event_id": r["event_id"], "type": r["type"],
                   "payload": json.loads(r["payload_json"]), "seq": r["seq"]}
                  for r in rows]
        if seq is not None:
            events = [e for e in events if e["seq"] <= seq]
        material = _material_from_nodes(nodes)
        revisions: dict[str, dict] = {}
        interpretation: dict[str, str] = {}
        nodes_by_id = {n["node_id"]: n for n in nodes}
        applied = []
        for event in events:
            _apply_event(material, revisions, interpretation, event, nodes_by_id)
            applied.append(event)
        snap = derive_snapshot(material, revisions, interpretation)
        snap["branch_id"] = branch_id
        snap["seq"] = events[-1]["seq"] if events else 0
        snap["material_hash"] = content_hash(material)
        snap["events"] = applied
        return snap

    def head_seq(self, conn, branch_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq),0) AS n FROM events WHERE branch_id=?",
            (branch_id,),
        ).fetchone()
        return int(row["n"])

    # ----- append --------------------------------------------------------
    def append_event(self, branch_id: str, etype: str, payload: dict,
                     expected_seq: int | None = None,
                     competing: dict | None = None) -> dict:
        """Append an event; ``expected_seq`` enforces optimistic concurrency."""
        with self.store.lock(), self.store.connect() as conn:
            if not self.store.branch_exists(branch_id):
                raise NotFoundError(f"unknown branch {branch_id}")
            actual = self.head_seq(conn, branch_id)
            if expected_seq is not None and expected_seq != actual:
                current_snap = self._snapshot_for_seq(conn, branch_id, actual)
                prior_snap = self._snapshot_for_seq(conn, branch_id, expected_seq)
                raise ConflictError(
                    "branch advanced since the view was loaded",
                    expected_seq, actual,
                    self._diff_snapshots(prior_snap, current_snap, competing),
                )
            seq = actual + 1
            event_id = payload.get("event_id") or f"E{branch_id}.{seq:03d}"
            payload = {**payload, "ts": _now()}
            self._materialize_event(conn, branch_id, seq, etype, payload)
            event_row = {
                "event_id": event_id, "type": etype, "payload": payload, "seq": seq,
            }
            digest = content_hash([branch_id, seq, etype, payload])
            conn.execute(
                "INSERT INTO events(seq, branch_id, event_id, type,"
                " payload_json, hash) VALUES(?,?,?,?,?,?)",
                (seq, branch_id, event_id, etype, canonical_json(payload), digest),
            )
            snap = self.replay(branch_id, seq)
            self.store._write_snapshot(conn, branch_id, seq, snap)
            return {"seq": seq, "event_id": event_id,
                    "snapshot_hash": content_hash(snap)}

    def _snapshot_for_seq(self, conn, branch_id: str, seq: int) -> dict:
        conn.commit()
        return self.replay(branch_id, seq)

    def _materialize_event(self, conn, branch_id: str, seq: int,
                           etype: str, payload: dict) -> None:
        """Create new immutable source bytes for source-amending events."""
        if etype != "source_amend":
            return
        nodes = self.store._read_nodes(conn, branch_id)
        material = _material_from_nodes(nodes)
        kind = payload["kind"]
        target_id = payload["target_id"]
        if kind == "segment":
            originals = material["segments"]
        elif kind == "channel":
            originals = material["channels"]
        else:
            originals = material["windows"]
        original = next((x for x in originals
                         if x["id"] == target_id or
                         x["id"] == target_id.split(":")[0]), None)
        if original is None:
            raise ValueError(f"cannot amend unknown {kind} {target_id}")
        new_content = copy.deepcopy(original)
        for field, value in payload["changes"].items():
            new_content[field] = value
        new_content["amended_by"] = payload.get("event_id", f"E{branch_id}.{seq:03d}")
        address = (f"{new_content['id']}:v{new_content.get('version', 1)}"
                   if kind == "window" else new_content["id"])
        node = {"kind": kind, "address": address, "content": new_content,
                "ordinal": 10_000 + seq}
        node["node_id"] = self._node_id(node)
        old = next((n for n in nodes if n["address"] == address), None)
        self._write_nodes(conn, branch_id, [node], 0,
                          old["node_id"] if old else None)
        payload["node_id"] = node["node_id"]

    # ----- branches and undo --------------------------------------------
    def fork(self, parent_branch: str, new_branch: str, title: str,
             at_seq: int | None = None) -> int:
        """Re-interpret the same source bytes on a new branch (no shared writes)."""
        with self.store.lock(), self.store.connect() as conn:
            if self.store.branch_exists(new_branch):
                raise ValueError(f"branch {new_branch} already exists")
            parent_nodes = self.store._read_nodes(conn, parent_branch)
            head = at_seq if at_seq is not None else self.head_seq(conn, parent_branch)
            self._write_nodes(conn, new_branch, parent_nodes, 0)
            rows = conn.execute(
                "SELECT * FROM events WHERE branch_id=? AND seq<=? ORDER BY seq",
                (parent_branch, head),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT INTO events(seq, branch_id, event_id, type,"
                    " payload_json, hash) VALUES(?,?,?,?,?,?)",
                    (row["seq"], new_branch, row["event_id"], row["type"],
                     row["payload_json"], row["hash"]),
                )
            conn.execute(
                "INSERT INTO branches(branch_id, material_id, parent_branch,"
                " parent_seq, title, created_seq, ordinal) VALUES(?,?,?,?,?,?,?)",
                (new_branch, self._material_id(conn, parent_branch),
                 parent_branch, head, title, 0,
                 self.store._next_branch_ordinal(conn)),
            )
            snap = self.replay(new_branch, head)
            self.store._write_snapshot(conn, new_branch, head, snap)
            return head

    def _material_id(self, conn, branch_id: str) -> str:
        row = conn.execute(
            "SELECT material_id FROM branches WHERE branch_id=?", (branch_id,)
        ).fetchone()
        return row["material_id"]

    def undo_ruling(self, branch_id: str, candidate_id: str) -> dict:
        """Append a reversal event; the earlier branch state stays readable."""
        snap = self.replay(branch_id)
        current = None
        for segs in snap["segments"].values():
            for seg in segs:
                for candidate in seg["candidates"]:
                    if candidate["id"] == candidate_id:
                        current = candidate["decision"]
        if current is None:
            raise NotFoundError(f"no ruling on {candidate_id}")
        return self.append_event(branch_id, "candidate_ruling", {
            "candidate_id": candidate_id, "decision": None,
            "reversal_of": snap["events"][-1]["event_id"] if snap["events"] else None,
            "note": "ruling undone; pending candidates restored",
        })

    def _diff_snapshots(self, old: dict, new: dict, competing: dict | None) -> dict:
        changed: list[dict] = []
        old_ids = {c["id"]: c for cid in old["segments"] for c in old["segments"][cid]}
        new_ids = {c["id"]: c for cid in new["segments"] for c in new["segments"][cid]}
        for sid, seg in new_ids.items():
            if sid not in old_ids:
                continue
            for metric in seg["window_metrics"]:
                prior = next((m for m in old_ids[sid]["window_metrics"]
                              if m["id"] == metric["id"]), None)
                if prior and prior["vpp"]["value"] != metric["vpp"]["value"]:
                    changed.append({"id": metric["id"],
                                    "yours": competing,
                                    "committed": {
                                        "vpp": metric["vpp"]["value"],
                                        "window_version": metric["window_version"],
                                        "by_event": metric["id"]}})
        return {
            "committed_after_your_view": [e["event_id"] for e in new["events"]
                                          if e["seq"] > old["seq"]],
            "changed_values": changed,
        }
