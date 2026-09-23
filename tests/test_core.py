"""领域核心单元测试：不等间隔时基、缺样/饱和插值、多峰候选、单位诊断。"""
import numpy as np
import pytest

import casepack
import core


def _baseline_seg(seg_id):
    data = casepack.load_baseline()
    return next(s for s in data["segments"] if s["segment_id"] == seg_id)


def test_nonuniform_timebase_no_index_subtraction():
    # 不等间隔时间轴：时间差必须由实际 t 逐点相减，而非索引*dt
    t = np.cumsum([0.0, 0.5, 0.7, 0.4, 0.6, 0.5, 0.8, 0.3])
    lag_s = core.index_lag_to_seconds(t, 3)
    assert lag_s == pytest.approx(float(np.mean(t[3:] - t[:-3])))
    assert lag_s != pytest.approx(3 * float(np.median(np.diff(t))))


def test_seg03_uses_actual_time_axis():
    norm = core.normalize_segment(_baseline_seg("seg-03"))
    assert not norm["uniform"]
    lag_s = core.index_lag_to_seconds(norm["t"], 4)
    naive = 4 * norm["dt_nominal"]
    assert abs(lag_s - naive) > 1e-15  # 索引相减的结果被否定


def test_missing_interval_not_interpolated():
    t = np.arange(10, dtype=float)
    v = np.arange(10, dtype=float)
    valid = np.ones(10, dtype=bool)
    valid[[4, 5, 6]] = False
    vals, excluded = core.interp_values(t, v, valid, [3.5, 5.0, 7.5])
    assert vals[0] == pytest.approx(3.5)      # 正常区间可插值
    assert np.isnan(vals[1]) and excluded[1]  # 缺样区间不进入插值
    assert vals[2] == pytest.approx(7.5)


def test_saturation_excluded_from_interpolation():
    seg = _baseline_seg("seg-04")
    norm = core.normalize_segment(seg)
    assert len(norm["saturated_idx"]) > 0
    assert not norm["valid"][norm["saturated_idx"]].any()
    t_mid = float(norm["t"][norm["saturated_idx"][len(norm["saturated_idx"]) // 2]])
    vals, excluded = core.interp_values(norm["t"], norm["v"], norm["valid"], [t_mid])
    assert np.isnan(vals[0]) and excluded[0]


def test_multi_peak_candidates_retained():
    data = casepack.load_baseline()
    segs = {s["segment_id"]: s for s in data["segments"]}
    ref = core.normalize_segment(segs["seg-01"])
    seg = core.normalize_segment(segs["seg-06"])
    cands = core.alignment_candidates(ref, seg)
    assert len(cands) >= 2  # 双脉冲重复事件 -> 多峰候选全部保留
    assert cands[0]["score"] >= cands[1]["score"]
    assert cands[1]["score"] >= 0.4 * cands[0]["score"]
    assert abs(abs(cands[1]["lag_s"]) - 16e-9) < 2e-9


def test_unconvertible_unit_stays_with_diagnostic():
    norm = core.normalize_segment(_baseline_seg("seg-07"), channel_deskew_s=25e-12)
    deskew_diag = [d for d in norm["diagnostics"] if d["field"] == "deskew"]
    assert deskew_diag, "不可转换的 deskew 必须留下诊断"
    assert deskew_diag[0]["raw"] == {"value": 5, "unit": "tick", "basis": "dac"}
    assert norm["deskew_s"] == pytest.approx(25e-12)  # 仅通道 deskew 生效


def test_to_seconds_units():
    assert core.to_seconds({"value": 25, "unit": "ps"}) == pytest.approx(25e-12)
    with pytest.raises(core.UnconvertibleUnit):
        core.to_seconds({"value": 5, "unit": "tick"})
