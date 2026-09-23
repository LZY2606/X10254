"""导出 / 导入档案：身份、顺序、来源关系在删除本地库再导入后仍保持。"""
from . import db as dbmod

TABLES = ["meta", "sources", "diagnostics", "branches", "events", "derivations"]
FORMAT = "waveforge-export"
ORDER = {"meta": "key", "sources": "ord", "diagnostics": "id",
         "branches": "created_gseq", "events": "gseq", "derivations": "id"}


def export_archive(conn):
    archive = {"format": FORMAT, "version": dbmod.SCHEMA_VERSION, "tables": {}}
    for table in TABLES:
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY {ORDER[table]}").fetchall()
        archive["tables"][table] = [dict(r) for r in rows]
    return archive


def import_archive(conn, archive):
    if archive.get("format") != FORMAT:
        raise ValueError(f"unknown archive format: {archive.get('format')}")
    if int(archive.get("version", 0)) > dbmod.SCHEMA_VERSION:
        raise ValueError("archive schema newer than this build")
    try:
        conn.execute("BEGIN")
        for table in reversed(TABLES):  # 先删子表（events/diagnostics）再删父表
            conn.execute(f"DELETE FROM {table}")
        for table in TABLES:
            for row in archive["tables"].get(table, []):
                cols = list(row.keys())
                conn.execute(
                    f"INSERT INTO {table}({','.join(cols)})"
                    f" VALUES({','.join('?' * len(cols))})",
                    [row[c] for c in cols])
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
            (str(dbmod.SCHEMA_VERSION),))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {t: len(archive["tables"].get(t, [])) for t in TABLES}
