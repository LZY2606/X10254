"""仓库层：原子导入、追加式分支事件、冲突检测与档案导出/导入。"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from station import db as dbmod
from station.ids import branch_id, canonical_json, event_id, material_id

MATERIAL_KINDS = ("samples", "timebase", "gain", "probe", "trigger", "deskew", "window")


class ConflictError(Exception):
    def __init__(self, expected_head: str | None, actual_head: str, diff: dict):
        super().__init__("版本冲突")
        self.expected_head = expected_head
        self.actual_head = actual_head
        self.diff = diff


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_sha(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload.get("raw_repr")).encode()).hexdigest()


def window_fields(window: dict) -> dict:
    """把测量区间版本展开进 material；原始 raw/单位/坐标基准逐字保留。"""
    return {
        "raw_repr": window["version"],
        "unit": "version",
        "coordinate_basis": "measurement_window_version",
        "version": window["version"],
        "extra": {
            "start_raw": window["start"]["raw_repr"],
            "start_unit": window["start"]["unit"],
            "start_basis": window["start"]["coordinate_basis"],
            "end_raw": window["end"]["raw_repr"],
            "end_unit": window["end"]["unit"],
            "end_basis": window["end"]["coordinate_basis"],
            "reason": window.get("reason"),
        },
    }


def material_records_from_case(case: dict) -> list[dict]:
    records: list[dict] = []
    for seg in case["segments"]:
        for kind in MATERIAL_KINDS:
            source_key = {"trigger": "trigger_ts", "window": "measurement_window"}.get(kind, kind)
            if kind == "window":
                payload = {"segment_no": seg["segment_no"], "kind": kind,
                           "channel": seg.get("channel"),
                           "label": seg.get("label")}
                payload.update(window_fields(seg["measurement_window"]))
                records.append(payload)
                continue
            source = seg[source_key]
            payload = {
                "segment_no": seg["segment_no"],
                "kind": kind,
                "channel": seg.get("channel"),
                "label": seg.get("label"),
                "version": None,
                "raw_repr": source.get("raw_repr"),
                "unit": source.get("unit"),
                "coordinate_basis": source.get("coordinate_basis"),
                "extra": {k: v for k, v in source.items()
                          if k not in ("raw_repr", "unit", "coordinate_basis",
                                       "version")},
            }
            records.append(payload)
    return records


def material_record_from_window(seg_no, channel, label, window: dict) -> dict:
    payload = {"segment_no": seg_no, "kind": "window", "channel": channel,
               "label": label}
    payload.update(window_fields(window))
    return payload


def upsert_material(conn, project_id: str, payload: dict) -> str:
    mid = material_id(payload)
    conn.execute(
        "INSERT INTO materials(material_id, project_id, segment_no, channel, kind, "
        "version_label, payload, raw_blob_sha256, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(material_id) DO NOTHING",
        (mid, project_id, payload["segment_no"], payload.get("channel"),
         payload["kind"], payload.get("version"), dbmod.jdumps(payload),
         _raw_sha(payload), now()),
    )
    return mid


def create_project(conn, name: str, case: dict) -> str:
    pid = "prj_" + hashlib.sha256(
        canonical_json({"name": name, "case_id": case["case_id"]}).encode()
    ).hexdigest()[:12]
    conn.execute(
        "INSERT OR IGNORE INTO projects(project_id, name, case_id, case_title, created_at) "
        "VALUES(?,?,?,?,?)",
        (pid, name, case["case_id"], case["title"], now()),
    )
    return pid


def get_or_create_project(conn, case: dict, name: str = "信号完整性审阅") -> str:
    row = conn.execute(
        "SELECT project_id FROM projects WHERE case_id=?", (case["case_id"],)
    ).fetchone()
    if row:
        return row["project_id"]
    return create_project(conn, name, case)


def branch_head(conn, bid: str) -> str | None:
    row = conn.execute(
        "SELECT event_id FROM events WHERE branch_id=? ORDER BY branch_seq DESC LIMIT 1",
        (bid,),
    ).fetchone()
    return row["event_id"] if row else None


def branch_next_seq(conn, bid: str, root_event_id: str | None) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(branch_seq),0) AS m FROM events WHERE branch_id=?",
        (bid,),
    ).fetchone()
    if row["m"]:
        return row["m"] + 1
    return 1


def create_branch(conn, project_id: str, name: str, parent_branch_id: str | None,
                  root_event_id: str | None) -> str:
    bid = branch_id(project_id, name, parent_branch_id, root_event_id)
    conn.execute(
        "INSERT OR IGNORE INTO branches(branch_id, project_id, name, parent_branch_id, "
        "root_event_id, created_at) VALUES(?,?,?,?,?,?)",
        (bid, project_id, name, parent_branch_id, root_event_id, now()),
    )
    return bid


def append_event(conn, project_id: str, bid: str, etype: str, payload: dict,
                 expected_head: str | None = None,
                 judgments: list[dict] | None = None,
                 check_head: bool = True) -> dict:
    root = conn.execute(
        "SELECT root_event_id FROM branches WHERE branch_id=?", (bid,)
    ).fetchone()["root_event_id"]
    actual_head = branch_head(conn, bid) or root
    if check_head and expected_head is not None and expected_head != actual_head:
        raise ConflictError(expected_head, actual_head, {
            "event_type": etype,
            "attempted_payload": payload,
            "current_head": _head_summary(conn, actual_head),
            "expected_head": _head_summary(conn, expected_head),
        })
    seq = branch_next_seq(conn, bid, root)
    eid = event_id(project_id, bid, seq, actual_head, etype, payload)
    conn.execute(
        "INSERT INTO events(event_id, project_id, branch_id, branch_seq, "
        "prev_event_id, type, payload, created_at) VALUES(?,?,?,?,?,?,?,?)",
        (eid, project_id, bid, seq, actual_head, etype,
         dbmod.jdumps(payload), now()),
    )
    for j in judgments or []:
        jid = "jud_" + hashlib.sha256(canonical_json(
            {"event": eid, "code": j["code"], "subject": j.get("subject"),
             "detail": j["detail"]}
        ).encode()).hexdigest()[:12]
        conn.execute(
            "INSERT OR IGNORE INTO judgments(judgment_id, event_id, branch_id, code, "
            "subject, detail, created_at) VALUES(?,?,?,?,?,?,?)",
            (jid, eid, bid, j["code"], j.get("subject"),
             dbmod.jdumps(j["detail"]), now()),
        )
    return {"event_id": eid, "seq": seq, "prev_event_id": actual_head}


def _head_summary(conn, eid: str | None) -> dict | None:
    if eid is None:
        return None
    row = conn.execute(
        "SELECT event_id, branch_seq, type, payload FROM events WHERE event_id=?",
        (eid,),
    ).fetchone()
    if not row:
        return {"event_id": eid, "missing": True}
    return {"event_id": eid, "seq": row["branch_seq"], "type": row["type"],
            "payload": dbmod.jloads(row["payload"])}


def import_case(conn, case: dict) -> dict:
    """整批原子导入；任何失败回滚，不留下半份记录。"""
    with dbmod.transaction(conn):
        pid = create_project(conn, case["title"], case)
        records = material_records_from_case(case)
        material_ids = []
        for rec in records:
            material_ids.append(upsert_material(conn, pid, rec))
        bid = create_branch(conn, pid, "main", None, None)
        # 若 main 已存在（重复导入同一案例），幂等返回
        if branch_head(conn, bid) is not None:
            return {"project_id": pid, "branch_id": bid, "idempotent": True}
        payload = {
            "case_id": case["case_id"],
            "case_title": case["title"],
            "rule_versions": case["rule_versions"],
            "materials": material_ids,
            "segment_count": len(case["segments"]),
        }
        head = append_event(conn, pid, bid, "import_batch", payload,
                            check_head=False)
        return {"project_id": pid, "branch_id": bid, "import_event": head,
                "material_count": len(material_ids)}


def import_case_failure_rolls_back(conn, case_bad: dict) -> None:
    """注入失败时必须回滚的演示/测试入口。"""
    before = conn.execute("SELECT COUNT(*) c FROM materials").fetchone()["c"]
    try:
        with dbmod.transaction(conn):
            pid = create_project(conn, "回滚测试", case_bad)
            for rec in material_records_from_case(case_bad):
                upsert_material(conn, pid, rec)
            raise RuntimeError("注入失败：模拟导入中断")
    except RuntimeError:
        pass
    after = conn.execute("SELECT COUNT(*) c FROM materials").fetchone()["c"]
    assert before == after


def store_derivation(conn, project_id: str, kind: str, subject: str,
                     rule_version: str, deps: list[str], body: dict) -> str:
    from station.ids import result_id

    rid = result_id(kind, subject, rule_version, sorted(set(deps)), body)
    conn.execute(
        "INSERT INTO derivations(result_id, project_id, kind, subject, rule_version, "
        "body, deps) VALUES(?,?,?,?,?,?,?) ON CONFLICT(result_id) DO NOTHING",
        (rid, project_id, kind, subject, rule_version, dbmod.jdumps(body),
         dbmod.jdumps(sorted(set(deps)))),
    )
    for dep in sorted(set(deps)):
        layer = "material" if dep.startswith("mat_") else "result"
        conn.execute(
            "INSERT INTO derivation_deps(from_id,to_result_id,layer) VALUES(?,?,?) "
            "ON CONFLICT(from_id,to_result_id) DO NOTHING",
            (dep, rid, layer),
        )
    return rid


def downstream_closure(conn, changed_ids: list[str]) -> list[str]:
    """依赖图反向闭包：直接或间接依赖被改材料的全部派生结果。"""
    seen: set[str] = set()
    frontier = list(changed_ids)
    while frontier:
        current = frontier.pop()
        rows = conn.execute(
            "SELECT to_result_id FROM derivation_deps WHERE from_id=?", (current,)
        ).fetchall()
        for row in rows:
            rid = row["to_result_id"]
            if rid not in seen:
                seen.add(rid)
                frontier.append(rid)
    return sorted(seen)


def save_trial(conn, trial_id: str, project_id: str, branch_id_v: str | None,
               name: str, mutated_source: str, adopted: bool,
               expected_scope: list[str], actual_scope: list[str],
               judgments: list[dict]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO counter_trials(trial_id, project_id, branch_id, name, "
        "mutated_source, adopted, expected_scope, actual_scope, judgments, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (trial_id, project_id, branch_id_v, name, mutated_source,
         1 if adopted else 0, dbmod.jdumps(expected_scope),
         dbmod.jdumps(actual_scope), dbmod.jdumps(judgments), now()),
    )


def export_archive(conn, project_id: str) -> dict:
    def rows(table):
        return [dict(r) for r in conn.execute(
            f"SELECT * FROM {table} WHERE project_id=?", (project_id,))]
    return {
        "archive_version": 1,
        "exported_at": now(),
        "projects": rows("projects"),
        "materials": rows("materials"),
        "derivations": rows("derivations"),
        "derivation_deps": [
            dict(r) for r in conn.execute(
                "SELECT d.* FROM derivation_deps d JOIN derivations r "
                "ON d.to_result_id=r.result_id WHERE r.project_id=?", (project_id,))
        ],
        "branches": rows("branches"),
        "events": rows("events"),
        "judgments": rows("judgments"),
        "counter_trials": rows("counter_trials"),
    }


def import_archive(conn, archive: dict) -> dict:
    """导入档案（典型流程：删除本地数据库后重建），身份逐字保持。"""
    with dbmod.transaction(conn):
        for table, cols in (
            ("projects", ("project_id", "name", "case_id", "case_title", "created_at")),
        ):
            for row in archive["projects"]:
                conn.execute(
                    f"INSERT OR REPLACE INTO {table}({','.join(cols)}) "
                    f"VALUES({','.join('?' * len(cols))})",
                    [row.get(c) for c in cols],
                )
        for row in archive["materials"]:
            conn.execute(
                "INSERT OR REPLACE INTO materials(material_id, project_id, segment_no, "
                "channel, kind, version_label, payload, raw_blob_sha256, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (row["material_id"], row["project_id"], row["segment_no"],
                 row["channel"], row["kind"], row["version_label"], row["payload"],
                 row["raw_blob_sha256"], row["created_at"]),
            )
        for row in archive["derivations"]:
            conn.execute(
                "INSERT OR REPLACE INTO derivations(result_id, project_id, kind, "
                "subject, rule_version, body, deps) VALUES(?,?,?,?,?,?,?)",
                (row["result_id"], row["project_id"], row["kind"], row["subject"],
                 row["rule_version"], row["body"], row["deps"]),
            )
        for row in archive["derivation_deps"]:
            conn.execute(
                "INSERT OR REPLACE INTO derivation_deps(from_id,to_result_id,layer) "
                "VALUES(?,?,?)", (row["from_id"], row["to_result_id"], row["layer"]))
        for row in archive["branches"]:
            conn.execute(
                "INSERT OR REPLACE INTO branches(branch_id, project_id, name, "
                "parent_branch_id, root_event_id, created_at) VALUES(?,?,?,?,?,?)",
                (row["branch_id"], row["project_id"], row["name"],
                 row["parent_branch_id"], row["root_event_id"], row["created_at"]),
            )
        for row in archive["events"]:
            conn.execute(
                "INSERT OR REPLACE INTO events(event_id, project_id, branch_id, "
                "branch_seq, prev_event_id, type, payload, created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (row["event_id"], row["project_id"], row["branch_id"],
                 row["branch_seq"], row["prev_event_id"], row["type"],
                 row["payload"], row["created_at"]),
            )
        for row in archive["judgments"]:
            conn.execute(
                "INSERT OR REPLACE INTO judgments(judgment_id, event_id, branch_id, "
                "code, subject, detail, created_at) VALUES(?,?,?,?,?,?)",
                (row["judgment_id"], row["event_id"], row["branch_id"], row["code"],
                 row["subject"], row["detail"], row["created_at"]),
            )
        for row in archive["counter_trials"]:
            conn.execute(
                "INSERT OR REPLACE INTO counter_trials(trial_id, project_id, branch_id, "
                "name, mutated_source, adopted, expected_scope, actual_scope, "
                "judgments, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (row["trial_id"], row["project_id"], row["branch_id"], row["name"],
                 row["mutated_source"], row["adopted"], row["expected_scope"],
                 row["actual_scope"], row["judgments"], row["created_at"]),
            )
        return {"projects": len(archive["projects"]),
                "materials": len(archive["materials"]),
                "events": len(archive["events"]),
                "branches": len(archive["branches"]),
                "derivations": len(archive["derivations"])}
