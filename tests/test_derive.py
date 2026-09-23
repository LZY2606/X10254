"""派生规则：R1 不等间隔不用索引相减 / R2 缺样饱和不插值 / R3 多峰候选。"""
import numpy as np

from waveforge import derive


def make_seg(seg_id, t, v, trigger, unit="mV", **kw):
    tb = ({"kind": "uniform", "t0": {"value": float(t[0]), "unit": "s"},
           "dt": {"value": float(t[1] - t[0]) * 1e12, "unit": "ps"}, "n": len(t)}
          if kw.pop("uniform", True) else
          {"kind": "nonuniform", "t": [float(x) for x in t], "t_unit": "s"})
    seg = {"id": seg_id, "channel": "ch1", "timebase": tb,
           "samples": [float(x) for x in v], "sample_unit": unit,
           "vertical_gain": {"value": 10, "unit": "mV/div"},
           "probe_attenuation": {"value": 1, "unit": "x"},
           "trigger": {"timestamp": {"value": trigger * 1e9, "unit": "ns"}},
           "deskew": {"value": 0, "unit": "ps"}}
    seg.update(kw)
    return seg


def pulse(t, te, amp=60.0):
    return amp * np.exp(-((t - te) / 2.5e-9) ** 2)


PARAMS = {"max_lag_s": 15e-9, "peak_threshold": 0.8}


def test_nonuniform_axis_uses_time_not_index():
    t_ref = 200e-12 * np.arange(512)
    ref = derive.build_segment_view(make_seg("seg:r", t_ref, pulse(t_ref, 50e-9), 50e-9))
    # 不等间隔轴：采样间隔随索引变化，索引差与时间差不成正比
    dt = 200e-12 * (1 + 0.05 * np.sin(np.arange(512) * 0.9))
    t_non = np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    true_lag = 50e-9 - 52.3e-9
    seg = derive.build_segment_view(
        make_seg("seg:n", t_non, pulse(t_non, 52.3e-9), 52.3e-9, uniform=False))
    assert any(e["rule"] == "nonuniform-no-index-sub" for e in seg["rule_events"])
    cands, _, _ = derive.correlation_candidates(ref, seg, PARAMS)
    assert cands and abs(cands[0]["tau"] - true_lag) < 200e-12


def test_missing_and_saturated_excluded_from_interp():
    t = 200e-12 * np.arange(512)
    v = pulse(t, 50e-9)
    v[100:120] = 80.0  # 超过满量程 -> 饱和
    seg = derive.build_segment_view(make_seg(
        "seg:g", t, v, 50e-9, missing=[[300, 20]],
        full_scale={"value": 80.0, "unit": "mV"}))
    assert not seg["valid"][100:120].any()   # 饱和
    assert not seg["valid"][300:320].any()   # 缺样
    assert any(e["rule"] == "gap-no-interp" for e in seg["rule_events"])
    gaps = derive.invalid_intervals(seg)
    assert any(lo <= t[310] <= hi for lo, hi in gaps)


def test_multipeak_candidates_all_kept():
    t = 200e-12 * np.arange(512)
    ref = derive.build_segment_view(make_seg("seg:r", t, pulse(t, 50e-9), 50e-9))
    two = pulse(t, 42e-9, 40) + pulse(t, 58e-9, 40)
    seg = derive.build_segment_view(make_seg("seg:m", t, two, 42e-9))
    cands, _, _ = derive.correlation_candidates(ref, seg, PARAMS)
    assert len(cands) >= 2  # R3：不收敛到单峰
    taus = sorted(c["tau"] for c in cands)
    assert abs(taus[0] - (-8e-9)) < 400e-12 and abs(taus[-1] - 8e-9) < 400e-12


def test_trigger_at_boundary_excluded():
    t = 200e-12 * np.arange(512)
    seg = derive.build_segment_view(make_seg("seg:b", t, pulse(t, 50e-9), 0.0))
    reasons = derive.exclusion_reasons(seg)
    assert reasons and reasons[0]["rule"] == "trigger-at-boundary"


def test_unconvertible_samples_stay_put():
    t = 200e-12 * np.arange(512)
    seg = derive.build_segment_view(make_seg("seg:u", t, pulse(t, 50e-9), 50e-9,
                                             unit="divs"))
    assert seg["v"] is None  # 不为了画图丢掉，也不伪造
    assert seg["raw"]["sample_unit"] == "divs"


def test_deskew_shifts_event_time(scenariod):
    from waveforge import store
    view = store.compute_branch_view(scenariod, "wave-overlay-demo", "br:edge-rules")
    ch2 = next(s for s in view["segments"] if s["id"] == "seg:ch2/0005")
    # deskew 350ps 已计入事件时刻（触发 50.6ns - deskew 0.35ns + tau）
    assert abs(view["jitter"]["seg:ch2/0005"]) < 1e-9
    assert ch2["selected"] is not None
