"""导入层：来源与解释分开保存。

- 每段原采样、时基、竖直增益、探头衰减、触发时标、deskew、测量区间版本化
  都以原始字节（raw JSON）进入 sources 表，导入时不做不可逆清洗。
- 单位/坐标基准在导入时只做“可转换性诊断”，无法转换的值留在原位置并附诊断。
- 导入失败整体回滚本次变更，不留下半份记录。
"""
import hashlib
import json
import os

from . import db as dbmod
from .units import to_si

SEGMENT_REQUIRED = ["id", "channel", "timebase", "samples", "sample_unit",
                    "vertical_gain", "probe_attenuation", "trigger", "deskew"]


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def diagnose_source(source_id, kind, raw_obj):
    """对单个来源做可转换性诊断，返回 diagnostics 列表。不修改 raw。"""
    diags = []

    def add(path, message, severity="warning"):
        diags.append({"id": f"diag:{source_id}:{len(diags)}", "source_id": source_id,
                      "path": path, "severity": severity, "message": message})

    if kind == "segment":
        for field in SEGMENT_REQUIRED:
            if field not in raw_obj:
                add(field, f"missing-required-field:{field}", "error")
        tb = raw_obj.get("timebase", {})
        if tb.get("kind") == "uniform":
            for f in ("t0", "dt"):
                _, err = to_si(tb.get(f), "time")
                if err:
                    add(f"timebase.{f}", err, "error")
        elif tb.get("kind") == "nonuniform":
            if "t" not in tb:
                add("timebase.t", "missing-required-field:timebase.t", "error")
            else:
                _, err = to_si({"value": 0, "unit": tb.get("t_unit", "")}, "time")
                if err:
                    add("timebase.t_unit", err, "error")
        else:
            add("timebase.kind", f"unknown-timebase-kind:{tb.get('kind')}", "error")
        _, err = to_si({"value": 0, "unit": raw_obj.get("sample_unit", "")}, "voltage")
        if err:
            add("sample_unit", err)
        for field, dim in (("vertical_gain", "gain"), ("probe_attenuation", "ratio"),
                           ("deskew", "time")):
            _, err = to_si(raw_obj.get(field), dim)
            if err:
                add(field, err)
        trig = raw_obj.get("trigger", {})
        _, err = to_si(trig.get("timestamp"), "time")
        if err:
            add("trigger.timestamp", err, "error")
    elif kind == "intervals":
        for i, win in enumerate(raw_obj.get("windows", [])):
            for f in ("t0", "t1"):
                _, err = to_si(win.get(f), "time")
                if err:
                    add(f"windows[{i}].{f}", err, "error")
    return diags


def insert_source(conn, case_id, kind, name, ord_, raw_text):
    src_id = f"src:{case_id}/{name}"
    raw_obj = json.loads(raw_text)
    conn.execute(
        "INSERT INTO sources(id, case_id, kind, name, ord, raw, sha256, imported_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (src_id, case_id, kind, name, ord_, raw_text, sha256_text(raw_text), dbmod.now()))
    for d in diagnose_source(src_id, kind, raw_obj):
        conn.execute(
            "INSERT INTO diagnostics(id, source_id, path, severity, message)"
            " VALUES(?,?,?,?,?)",
            (d["id"], d["source_id"], d["path"], d["severity"], d["message"]))
    return src_id


def load_case_package(conn, pkg_dir):
    """导入案例包基线。事务性：任一来源失败则回滚本次全部变更。"""
    manifest_path = os.path.join(pkg_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    case_id = manifest["case_id"]
    existing = conn.execute(
        "SELECT COUNT(*) c FROM sources WHERE case_id=?", (case_id,)).fetchone()["c"]
    if existing:
        raise CaseAlreadyLoaded(case_id)
    try:
        conn.execute("BEGIN")
        for ord_, entry in enumerate(manifest["sources"]):
            path = os.path.join(pkg_dir, entry["file"])
            with open(path, encoding="utf-8") as fh:
                raw_text = fh.read()
            # 校验 JSON 可解析；失败则整体回滚
            json.loads(raw_text)
            insert_source(conn, case_id, entry["kind"], entry["name"], ord_, raw_text)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {"case_id": case_id, "title": manifest.get("title"),
            "sources": len(manifest["sources"])}


class CaseAlreadyLoaded(Exception):
    pass


def get_sources(conn, case_id=None, kind=None):
    sql = "SELECT * FROM sources"
    cond, args = [], []
    if case_id:
        cond.append("case_id=?"); args.append(case_id)
    if kind:
        cond.append("kind=?"); args.append(kind)
    if cond:
        sql += " WHERE " + " AND ".join(cond)
    sql += " ORDER BY ord"
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def get_diagnostics(conn, source_id=None):
    if source_id:
        rows = conn.execute(
            "SELECT * FROM diagnostics WHERE source_id=? ORDER BY id", (source_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM diagnostics ORDER BY id").fetchall()
    return [dict(r) for r in rows]
