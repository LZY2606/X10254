"""Branch state replay and dependency-scoped derived-cache maintenance.

Sources (raw segment bytes) and interpretations (events, adjudications,
parameter/window versions) are stored separately. Re-interpreting the same
material creates a new branch; old branches, adjudications and the raw bytes
they referenced remain readable.
"""
import json

import db
import derive


def chain_events(conn, branch_id):
    """Full event chain for a branch: parent chain up to the base event,
    then the branch's own events, in replay order."""
    br = conn.execute("SELECT * FROM branches WHERE id=?", (branch_id,)).fetchone()
    if br is None:
        raise KeyError("branch not found: %s" % branch_id)
    chain = []
    if br["parent_id"]:
        parent = dict(conn.execute(
            "SELECT * FROM branches WHERE id=?", (br["parent_id"],)).fetchone())
        parent_chain = chain_events(conn, parent["id"])
        base_seq = None
        if br["base_event_id"]:
            for e in parent_chain:
                if e["id"] == br["base_event_id"]:
                    base_seq = e["_chain_idx"]
                    break
        chain.extend(e for e in parent_chain
                     if base_seq is None or e["_chain_idx"] <= base_seq)
    own = db.list_events(conn, branch_id)
    for i, e in enumerate(chain):
        e["_chain_idx"] = i
    offset = len(chain)
    for j, e in enumerate(own):
        e["_chain_idx"] = offset + j
    return chain + own


def _deep_merge(dst, patch):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def replay(conn, branch_id, upto_event_id=None):
    """Rebuild branch state from the event log.

    Undo events negate their target without rewriting history; viewing at an
    old event restores the dependency versions and visible conclusions of
    that moment rather than overlaying the latest computation.
    """
    chain = chain_events(conn, branch_id)
    if upto_event_id is not None:
        cut = None
        for i, e in enumerate(chain):
            if e["id"] == upto_event_id:
                cut = i
                break
        if cut is None:
            raise KeyError("event not in branch chain: %s" % upto_event_id)
        chain = chain[: cut + 1]

    case_meta = db.get_meta(conn, "case_meta", {})
    state = {
        "params": dict(case_meta.get("params", {})),
        "mw_version": case_meta.get("initial_mw_version"),
        "segments": {},
        "order": [],
        "adjudications": {},
        "applied": [],
    }
    undone = {e["payload"].get("target") for e in chain if e["kind"] == "undo"}
    for e in chain:
        if e["id"] in undone or e["kind"] == "undo":
            continue
        p = e["payload"]
        if e["kind"] == "import":
            rows = conn.execute(
                "SELECT * FROM segments WHERE import_id=? ORDER BY ordinal",
                (p["import_id"],)).fetchall()
            for r in rows:
                state["segments"][r["id"]] = {
                    "id": r["id"], "key": r["seg_key"], "channel": r["channel"],
                    "raw": json.loads(r["raw_json"]), "hash": r["source_hash"],
                    "diagnostics": json.loads(r["diagnostics"] or "[]"),
                }
                state["order"].append(r["id"])
        elif e["kind"] == "param_revision":
            state["params"][p["key"]] = p["to"]
        elif e["kind"] == "window_revision":
            state["mw_version"] = p["to"]
        elif e["kind"] == "source_patch":
            seg = state["segments"].get(p["segment_id"])
            if seg is not None:
                _deep_merge(seg["raw"], p["patch"])
                seg["hash"] = derive.source_hash(seg["raw"])
        elif e["kind"] == "adjudicate":
            state["adjudications"][p["candidate_id"]] = {
                "decision": p["decision"],
                "rationale": p.get("rationale", ""),
                "event_id": e["id"],
            }
        state["applied"].append(e["id"])
    return state, chain, case_meta


def derived_items(derived):
    """Flatten the derived layer into (item_id, kind, subject, deps, payload)."""
    items = {}
    for sid, a in derived["alignment"].items():
        items[a["id"]] = ("alignment", a["segment"], a["deps"], a)
    for sid, j in derived["jitter"].items():
        items[j["id"]] = ("jitter", j["segment"], j["deps"], j)
    for sid, w in derived["windows"].items():
        items[w["id"]] = ("window", w["segment"], w["deps"], w)
    for ex in derived["exclusions"]:
        items[ex["id"]] = ("exclusion", ex["segment"], ex.get("deps", []), ex)
    for d in derived["diagnostics"]:
        items[d["id"]] = ("diagnostic", d["segment"], [], d)
    return items


def refresh_derived(conn, branch_id):
    """Recompute the derived layer at branch head and compute the
    invalidation scope from dependency versions: only items whose deps or
    payload changed are superseded; old rows stay readable (valid=0)."""
    state, chain, case_meta = replay(conn, branch_id)
    derived = derive.compute_derived(state, case_meta)
    items = derived_items(derived)
    existing = conn.execute(
        "SELECT * FROM derived WHERE branch_id=? AND valid=1",
        (branch_id,)).fetchall()
    existing = {r["id"]: r for r in existing}
    for iid, (kind, subject, deps, payload) in items.items():
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        deps_blob = json.dumps(deps, sort_keys=True)
        old = existing.pop(iid, None)
        if old and old["payload"] == blob and old["deps"] == deps_blob:
            continue  # unchanged: outside the invalidation scope
        if old:
            conn.execute("UPDATE derived SET valid=0 WHERE id=? AND branch_id=? AND valid=1",
                         (iid, branch_id))
        conn.execute(
            "INSERT INTO derived(id,branch_id,kind,subject_id,deps,payload,valid,computed_at)"
            " VALUES(?,?,?,?,?,?,1,?)",
            (iid, branch_id, kind, subject, deps_blob, blob, db.now()))
    for iid, old in existing.items():
        conn.execute("UPDATE derived SET valid=0 WHERE id=? AND branch_id=? AND valid=1",
                     (iid, branch_id))
    conn.commit()
    return state, derived, case_meta


def invalidation_scope(conn, branch_id):
    rows = conn.execute(
        "SELECT * FROM derived WHERE branch_id=? AND valid=0 ORDER BY computed_at",
        (branch_id,)).fetchall()
    return [{"id": r["id"], "kind": r["kind"], "subject": r["subject_id"],
             "deps": json.loads(r["deps"]), "superseded_at": r["computed_at"]}
            for r in rows]
