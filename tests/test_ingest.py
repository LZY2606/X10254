"""导入层：来源与解释分离、原始表示保留、失败整体回滚。"""
import json
import os
import shutil

import pytest

from waveforge import ingest
from waveforge.ingest import get_diagnostics, get_sources

from conftest import CASE, PKG


def test_baseline_sources_and_raw_preserved(loaded):
    srcs = get_sources(loaded, case_id=CASE)
    assert len(srcs) == 12
    kinds = {s["kind"] for s in srcs}
    assert kinds == {"segment", "segment-patch", "intervals"}
    # 原始字节可读回且哈希稳定
    seg = next(s for s in srcs if s["name"] == "segments/ch1_0002")
    raw = json.loads(seg["raw"])
    assert raw["timebase"]["kind"] == "nonuniform"
    assert raw["sample_unit"] == "mV"  # 原始表示保留，导入时不规范化
    assert seg["sha256"] == ingest.sha256_text(seg["raw"])
    # 顺序稳定
    assert [s["ord"] for s in srcs] == sorted(s["ord"] for s in srcs)


def test_unconvertible_unit_kept_with_diagnostic(loaded):
    diags = get_diagnostics(loaded)
    hit = [d for d in diags if "ch1_0008" in d["source_id"] and d["path"] == "sample_unit"]
    assert hit and "unconvertible-unit:divs" in hit[0]["message"]
    # 原值仍在原位置
    src = next(s for s in get_sources(loaded) if "ch1_0008" in s["id"])
    assert json.loads(src["raw"])["sample_unit"] == "divs"


def test_failed_import_rolls_back(conn, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(PKG, broken)
    # 破坏一个来源文件
    with open(broken / "segments" / "ch1_0003.json", "w") as fh:
        fh.write("{not json")
    with pytest.raises(Exception):
        ingest.load_case_package(conn, str(broken))
    assert get_sources(conn) == []  # 不留半份记录
    assert get_diagnostics(conn) == []


def test_double_load_rejected(loaded):
    with pytest.raises(ingest.CaseAlreadyLoaded):
        ingest.load_case_package(loaded, PKG)
