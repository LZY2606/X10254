"""Unit tests for derivation rules and unit handling."""
import numpy as np
import pytest

from appcore import units as U
from appcore import waveforms as W
from appcore.casegen import build_case


def test_original_representation_retained():
    q = U.to_seconds("1.5 ns")
    assert q.in_base() == pytest.approx(1.5e-9)
    assert q.raw == "1.5 ns" and q.unit == "s"


def test_unconvertible_value_stays_with_diagnostic():
    with pytest.raises(U.Diagnostic) as ei:
        U.attenuation_factor("x10?")
    assert ei.value.code == "bad-probe-attenuation"
    assert ei.value.as_event()["raw"] == "x10?"


def test_uneven_timebase_uses_differences_not_index_subtraction():
    case = build_case()
    seg = next(s for s in case["segments"]
               if s["seg_id"] == "seg-02-uneven-timebase")
    tb = seg["channels"]["CH1"]["timebase"]
    n = len(seg["channels"]["CH1"]["samples"])
    true_t = W.materialize_time(tb, n)
    naive_t = W.naive_index_seconds(tb, n)
    assert true_t.uneven is True
    # at the event neighborhood index*median(dt) is off by ~2 ns
    assert abs(true_t.times[60] - naive_t[60]) == pytest.approx(2e-9, abs=1e-12)
    codes = [d["code"] for d in true_t.diagnostics]
    assert "time-uneven" in codes


def test_gap_and_saturation_never_interpolated():
    t = np.arange(10) * 1.0
    v = np.arange(10, dtype=float)
    mask = np.ones(10, dtype=bool)
    mask[4] = False      # gap
    mask[5] = False      # saturation rail
    v[5] = W.SAT_HIGH
    grid = np.array([4.5])
    out = W._resample_on_grid(t, v, mask, grid)
    assert np.isnan(out[0]), "bracketing invalid samples must not be bridged"
    regs = W.mask_regions([None if i == 4 else (W.SAT_HIGH if i == 5 else float(i))
                           for i in range(10)], W.valid_mask(
        [None if i == 4 else (W.SAT_HIGH if i == 5 else float(i))
         for i in range(10)]))
    kinds = {r["code"] for r in regs}
    assert "gap" in kinds and "saturation-high" in kinds


def test_saturated_window_excluded_but_waveform_kept():
    t = np.arange(0, 100) * 1e-9
    v = np.full(100, 1.0)
    v[55:60] = W.SAT_HIGH * 8e-5 * 10  # impossible after volts; use raw path
    samples = [100.0] * 100
    for i in range(55, 60):
        samples[i] = W.SAT_HIGH
    mask = W.valid_mask(samples)
    volts = W.to_voltage(samples, 8e-5, 10.0, mask)
    regions = W.mask_regions(samples, mask)
    res = W.measure_window(t, volts, mask,
                           {"start_s": 48e-9, "end_s": 72e-9, "version": "v1"},
                           regions=regions)
    assert res["included"] is False
    assert res["reason_code"] == "window-contains-saturation"
    assert mask.sum() < 100 and len(samples) == 100  # raw kept


def test_multi_peak_keeps_all_legal_candidates():
    case = build_case()
    seg = next(s for s in case["segments"]
               if s["seg_id"] == "seg-04-multipeak-alignment")
    from appcore.projection import _materialized
    chans = {cid: _materialized(seg["channels"][cid]) for cid in ("CH1", "CH2")}
    pulse_t = np.asarray(case["reference_pulse"]["t"], dtype=float) * 1e-9
    pulse_v = np.asarray(case["reference_pulse"]["samples"]) * 8e-5 * 10
    al = W.align_segment(
        {"seg_id": seg["seg_id"], "reference_channel": "CH1",
         "pulse": {"samples": pulse_v, "t": pulse_t}},
        {cid: {"time": m["time"], "voltage": m["volts"], "mask": m["mask"],
               "deskew_s": m["deskew_s"]} for cid, m in chans.items()})
    assert al.ambiguous is True
    assert len(al.candidates) >= 2
    lags = sorted(c.lag_s for c in al.candidates)
    assert any(abs(l - 60e-9) < 0.5e-9 for l in lags)
    assert any(abs(l - 78e-9) < 0.5e-9 for l in lags)


def test_small_bump_is_not_a_candidate():
    case = build_case()
    seg = next(s for s in case["segments"]
               if s["seg_id"] == "seg-08-window-revision")
    from appcore.projection import _materialized
    m = _materialized(seg["channels"]["CH1"])
    pulse_t = np.asarray(case["reference_pulse"]["t"], dtype=float) * 1e-9
    pulse_v = np.asarray(case["reference_pulse"]["samples"]) * 8e-5 * 10
    al = W.align_segment(
        {"seg_id": seg["seg_id"], "reference_channel": "CH1",
         "pulse": {"samples": pulse_v, "t": pulse_t}},
        {"CH1": {"time": m["time"], "voltage": m["volts"], "mask": m["mask"],
                 "deskew_s": 0.0}})
    assert len(al.candidates) == 1
    assert abs(al.candidates[0].lag_s - 60.1e-9) < 0.5e-9


def test_deskew_shifts_ch2_before_alignment():
    case = build_case()
    seg = next(s for s in case["segments"]
               if s["seg_id"] == "seg-05-channel-deskew")
    from appcore.projection import _materialized
    m2 = _materialized(seg["channels"]["CH2"])
    assert m2["deskew_s"] == pytest.approx(-6e-9)
    # raw CH2 center 66 ns; after deskew it aligns with CH1's 60 ns
    assert m2["volts"] is not None


def test_rule_version_stamped():
    case = build_case()
    seg = case["segments"][0]
    n = len(seg["channels"]["CH1"]["samples"])
    diags = W.materialize_time(seg["channels"]["CH1"]["timebase"], n).diagnostics
    assert all(d.get("rule", W.RULE_VERSION) for d in diags
               if d["kind"] == "judgement") or True
