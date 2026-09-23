"""状态迁移：v1 旧库升级到当前版本，数据不丢。"""
import os

from waveforge import db as dbmod


def make_v1_db(path):
    conn = dbmod.connect(path)
    conn.executescript(dbmod.DDL_V1)
    conn.execute("INSERT INTO meta VALUES('schema_version','1')")
    conn.execute("INSERT INTO sources VALUES('src:c/s1','c','segment','s1',0,"
                 "'{}','abc',1.0)")
    conn.execute("INSERT INTO branches VALUES('br:main','main',NULL,NULL,1)")
    conn.execute("INSERT INTO events(id,branch_id,seq,kind,payload,rule_version,"
                 "params_version,interval_version,created_at) VALUES"
                 "('evt:br:main/0001','br:main',1,'import-baseline','{}',"
                 "'align-rules:v1','params:v1','intervals:v1',1.0)")
    conn.commit()
    conn.close()


def test_migrate_v1_to_current(tmp_path):
    path = str(tmp_path / "old.db")
    make_v1_db(path)
    conn = dbmod.connect(path)
    applied = dbmod.migrate(conn)
    assert applied == [2]
    assert dbmod.schema_version(conn) == dbmod.SCHEMA_VERSION
    # 旧数据保留
    assert conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"] == 1
    row = conn.execute("SELECT * FROM events WHERE id='evt:br:main/0001'").fetchone()
    assert row["undone"] == 0  # 新列有默认值
    # 新表可用
    conn.execute("INSERT INTO diagnostics VALUES('d1','src:c/s1','p','warning','m')")
    assert conn.execute("SELECT COUNT(*) c FROM diagnostics").fetchone()["c"] == 1
    # 迁移幂等
    assert dbmod.migrate(conn) == []
