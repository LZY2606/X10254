"""SQLite event-sourced repository.

Source material is immutable. Re-interpretation appends events on a branch;
old branches, rulings and the raw bytes they reference stay readable. Every
identity is stable across export -> delete -> import because ids are content-
and-path-derived (never rowids / AUTOINCREMENT).
"""
from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS branch (
  branch_id   TEXT PRIMARY KEY,
  parent_id   TEXT,
  forked_from_event TEXT,
  label       TEXT NOT NULL,
  created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS event (
  event_id    TEXT PRIMARY KEY,
  branch_id   TEXT NOT NULL,
  seq         INTEGER NOT NULL,
  type        TEXT NOT NULL,
  payload     TEXT NOT NULL,
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE(branch_id, seq)
);
CREATE TABLE IF NOT EXISTS blob (
  blob_id     TEXT PRIMARY KEY,
  content     BLOB NOT NULL,
  sha256      TEXT NOT NULL,
  media_type  TEXT
);
CREATE TABLE IF NOT EXISTS material (
  material_id TEXT PRIMARY KEY,
  slot        TEXT NOT NULL,
  blob_id     TEXT REFERENCES blob(blob_id),
  json_value  TEXT,
  supersedes  TEXT
);
CREATE TABLE IF NOT EXISTS derived_cache (
  cache_id    TEXT PRIMARY KEY,
  event_id    TEXT NOT NULL,
  kind        TEXT NOT NULL,
  ident       TEXT NOT NULL,
  value       TEXT NOT NULL,
  UNIQUE(event_id, kind, ident)
);
CREATE INDEX IF NOT EXISTS idx_event_branch ON event(branch_id);
CREATE INDEX IF NOT EXISTS idx_cache_event ON derived_cache(event_id);
"""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _b64e(content: bytes) -> str:
    return base64.b64encode(content).decode()


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode())


def stable_id(prefix: str, *parts: Any) -> str:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return f"{prefix}_{h[:16]}"


class Conflict(Exception):
    """Optimistic append conflict (two windows saved from the same base)."""

    def __init__(self, message: str, server_event_id: str,
                 your_event_id: str, diff: dict):
        super().__init__(message)
        self.message = message
        self.server_event_id = server_event_id
        self.your_event_id = your_event_id
        self.diff = diff


class Repository:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        self.conn.executescript(_SCHEMA)
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),))
            self.conn.commit()
            return
        version = int(row["value"])
        if version < SCHEMA_VERSION:
            # v1 -> v2: derived result cache table (created by _SCHEMA above)
            # v2 -> v3: cache lookup index
            self.conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),))
            self.conn.commit()

    def schema_version(self) -> int:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row["value"]) if row else 0

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ----- branches --------------------------------------------------------
    def ensure_main_branch(self) -> str:
        row = self.conn.execute(
            "SELECT branch_id FROM branch WHERE branch_id='main'").fetchone()
        if row:
            return "main"
        with self.transaction() as c:
            c.execute("INSERT INTO branch(branch_id,parent_id,"
                      "forked_from_event,label,created_seq) "
                      "VALUES('main',NULL,NULL,'main',0)")
        return "main"

    def create_branch(self, parent_id: str, forked_from_event: str,
                      label: str) -> str:
        branch_id = stable_id("branch", parent_id, forked_from_event, label)
        seq = self.next_global_seq()
        with self.transaction() as c:
            c.execute("INSERT INTO branch(branch_id,parent_id,"
                      "forked_from_event,label,created_seq) VALUES(?,?,?,?,?)",
                      (branch_id, parent_id, forked_from_event, label, seq))
        return branch_id

    def list_branches(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branch ORDER BY created_seq").fetchall()]

    def _event_seq(self, event_id: str | None) -> int | None:
        if not event_id:
            return None
        row = self.conn.execute(
            "SELECT seq FROM event WHERE event_id=?", (event_id,)).fetchone()
        return int(row["seq"]) if row else None

    def branch_events(self, branch_id: str) -> list[dict]:
        """Events visible on a branch: ancestry chain, truncated at forks."""
        chain: list[dict] = []
        current = branch_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            row = self.conn.execute(
                "SELECT * FROM branch WHERE branch_id=?", (current,)).fetchone()
            if not row:
                break
            chain.append(dict(row))
            current = row["parent_id"]
        if not chain:
            return []
        # cap[branch] = max seq visible from that ancestor
        caps: dict[str, int | None] = {b["branch_id"]: None for b in chain}
        for i in range(len(chain) - 1):
            child, parent = chain[i], chain[i + 1]
            if child["forked_from_event"]:
                s = self._event_seq(child["forked_from_event"])
                if s is not None:
                    pid = parent["branch_id"]
                    caps[pid] = s if caps[pid] is None else min(caps[pid], s)
        out: list[dict] = []
        for b in chain:
            cap = caps[b["branch_id"]]
            rows = self.conn.execute(
                "SELECT * FROM event WHERE branch_id=? ORDER BY seq",
                (b["branch_id"],)).fetchall()
            for r in rows:
                if cap is None or r["seq"] <= cap:
                    out.append(dict(r))
        out.sort(key=lambda r: (r["seq"], r["event_id"]))
        return out

    # ----- events ----------------------------------------------------------
    def next_global_seq(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS s FROM event").fetchone()
        return int(row["s"])

    def append(self, branch_id: str, etype: str, payload: dict,
               event_id: str | None = None,
               expected_last_event: str | None = None,
               conflict_diff: dict | None = None) -> dict:
        seq = self.next_global_seq()
        eid = event_id or stable_id("evt", branch_id, seq, etype,
                                    json.dumps(payload, sort_keys=True))
        last = self.conn.execute(
            "SELECT event_id,payload FROM event WHERE branch_id=? "
            "ORDER BY seq DESC LIMIT 1", (branch_id,)).fetchone()
        if expected_last_event is not None:
            current_tip = last["event_id"] if last else None
            if current_tip != expected_last_event:
                raise Conflict(
                    f"branch {branch_id} moved since your last read",
                    server_event_id=current_tip or "",
                    your_event_id=eid,
                    diff=conflict_diff or {
                        "type": etype,
                        "server_tip": current_tip,
                        "client_base": expected_last_event,
                        "server_payload": json.loads(last["payload"]) if last else {},
                        "your_payload": payload,
                    })
        with self.transaction() as c:
            c.execute("INSERT INTO event(event_id,branch_id,seq,type,payload) "
                      "VALUES(?,?,?,?,?)",
                      (eid, branch_id, seq, etype,
                       json.dumps(payload, sort_keys=True, ensure_ascii=False)))
        return {"event_id": eid, "seq": seq, "branch_id": branch_id,
                "type": etype, "payload": payload}

    def get_event(self, event_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM event WHERE event_id=?", (event_id,)).fetchone()
        return dict(row) if row else None

    def all_events(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM event ORDER BY seq").fetchall()]

    # ----- materials / blobs ----------------------------------------------
    def put_blob(self, content: bytes,
                 media_type: str = "application/json") -> str:
        bid = stable_id("blob", _sha256(content))
        with self.transaction() as c:
            c.execute("INSERT OR IGNORE INTO blob(blob_id,content,sha256,"
                      "media_type) VALUES(?,?,?,?)",
                      (bid, content, _sha256(content), media_type))
        return bid

    def put_material(self, slot: str, value: Any,
                     supersedes: str | None = None) -> str:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        bid = self.put_blob(raw)
        mid = stable_id("mat", slot, _sha256(raw))
        with self.transaction() as c:
            c.execute("INSERT OR IGNORE INTO material(material_id,slot,"
                      "blob_id,json_value,supersedes) VALUES(?,?,?,?,?)",
                      (mid, slot, bid,
                       json.dumps(value, ensure_ascii=False), supersedes))
        return mid

    def get_material(self, material_id: str) -> Any:
        row = self.conn.execute(
            "SELECT json_value FROM material WHERE material_id=?",
            (material_id,)).fetchone()
        return json.loads(row["json_value"]) if row else None

    def list_materials(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT material_id,slot,supersedes FROM material "
            "ORDER BY material_id").fetchall()]

    # ----- derived cache ---------------------------------------------------
    def cache_derived(self, event_id: str, kind: str, ident: str,
                      value: Any) -> None:
        cid = stable_id("der", event_id, kind, ident)
        with self.transaction() as c:
            c.execute("INSERT OR REPLACE INTO derived_cache(cache_id,event_id,"
                      "kind,ident,value) VALUES(?,?,?,?,?)",
                      (cid, event_id, kind, ident,
                       json.dumps(value, ensure_ascii=False)))

    def cached_at(self, event_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT kind,ident,value FROM derived_cache WHERE event_id=? "
            "ORDER BY kind,ident", (event_id,)).fetchall()]

    # ----- export / import (identities preserved) -------------------------
    def export_bundle(self) -> dict:
        events = [dict(r) for r in self.conn.execute(
            "SELECT event_id,branch_id,seq,type,payload,created_at FROM event "
            "ORDER BY seq").fetchall()]
        branches = self.list_branches()
        materials = [dict(r) for r in self.conn.execute(
            "SELECT material_id,slot,blob_id,json_value,supersedes "
            "FROM material ORDER BY material_id").fetchall()]
        blobs = []
        for r in self.conn.execute(
                "SELECT blob_id,content,sha256,media_type FROM blob"
                ).fetchall():
            blobs.append({"blob_id": r["blob_id"],
                          "content_b64": _b64e(r["content"]),
                          "sha256": r["sha256"], "media_type": r["media_type"]})
        return {"format": "waveform-lineage/1",
                "rule_version": __import__("appcore.waveforms", fromlist=["x"]).RULE_VERSION,
                "schema_version": self.schema_version(),
                "branches": branches, "events": events,
                "materials": materials, "blobs": blobs}

    def import_bundle(self, bundle: dict) -> dict:
        """Atomic import; any failure rolls the whole change back."""
        if bundle.get("format") != "waveform-lineage/1":
            raise ValueError("unsupported bundle format")
        if self.all_events():
            raise ValueError("repository is not empty; refusing to merge ids")
        blobs = bundle.get("blobs", [])
        materials = bundle.get("materials", [])
        branches = bundle.get("branches", [])
        events = bundle.get("events", [])
        # validate ids / checksums before writing anything
        for b in blobs:
            content = _b64d(b["content_b64"])
            if _sha256(content) != b["sha256"]:
                raise ValueError(f"blob checksum mismatch: {b['blob_id']}")
        try:
            with self.transaction() as c:
                for b in blobs:
                    content = _b64d(b["content_b64"])
                    c.execute("INSERT INTO blob(blob_id,content,sha256,"
                              "media_type) VALUES(?,?,?,?)",
                              (b["blob_id"], content, b["sha256"],
                               b.get("media_type")))
                for m in materials:
                    c.execute("INSERT INTO material(material_id,slot,blob_id,"
                              "json_value,supersedes) VALUES(?,?,?,?,?)",
                              (m["material_id"], m["slot"], m["blob_id"],
                               m["json_value"], m.get("supersedes")))
                for br in branches:
                    c.execute("INSERT INTO branch(branch_id,parent_id,"
                              "forked_from_event,label,created_seq) "
                              "VALUES(?,?,?,?,?)",
                              (br["branch_id"], br.get("parent_id"),
                               br.get("forked_from_event"), br["label"],
                               br["created_seq"]))
                for e in events:
                    c.execute("INSERT INTO event(event_id,branch_id,seq,type,"
                              "payload,created_at) VALUES(?,?,?,?,?,?)",
                              (e["event_id"], e["branch_id"], e["seq"],
                               e["type"], e["payload"],
                               e.get("created_at",
                                     "2026-01-01T00:00:00.000Z")))
        except Exception:
            raise
        return {"events": len(events), "materials": len(materials),
                "branches": len(branches), "blobs": len(blobs)}
