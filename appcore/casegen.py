"""Bundled offline domain case: segmented capture with trigger jitter.

Every source value keeps its original representation (strings with units,
``None`` gaps, rail codes). Normalization happens only in the derivation layer.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

N = 160          # samples per segment on the uniform base
DT_NS = 1.0      # nominal sample interval
TRIG_I = 50      # nominal trigger index
CENTER_NS = 60.0 # nominal event center (absolute segment time)
SIGMA_NS = 3.0
AMP = 2000.0


def _pulse(t: np.ndarray, center: float, amp: float = AMP,
           sigma: float = SIGMA_NS) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _uniform_channel(center: float, *, amp: float = AMP, clip: bool = False,
                     gaps: tuple[int, int] | None = None,
                     second: tuple[float, float] | None = None) -> list:
    t = np.arange(N) * DT_NS
    v = _pulse(t, center, amp)
    if second:
        v = v + _pulse(t, second[0], second[1])
    if clip:
        v = np.minimum(v, 32767.0)
    out = [None] * N
    for i, x in enumerate(v):
        out[i] = None
    for i, x in enumerate(v):
        out[i] = float(round(x, 3))
    if gaps:
        for i in range(gaps[0], gaps[1]):
            out[i] = None
    if clip:
        for i, x in enumerate(v):
            if x >= 32767.0:
                out[i] = 32767
    return out


def _uneven_axis(center: float) -> tuple[list, list, dict]:
    pattern = [1.3, 1.0, 0.6]
    dts = np.array([pattern[i % 3] for i in range(N - 1)])
    t = np.concatenate([[0.0], np.cumsum(dts)])
    v = _pulse(t, center)
    return [float(round(x, 3)) for x in v], [float(round(x, 6)) for x in t], \
        {"unit": "ns", "t": [float(round(x, 6)) for x in t]}


def _ch(ch1_samples, ch2_samples, *, timebase=None, trig_index=TRIG_I,
        trig_time="50 ns", ch1_gain="0.08 mV/count", ch2_gain="0.08 mV/count",
        ch1_probe="10x", ch2_probe="10x", ch2_deskew="-6 ns",
        ch2_center_shift=6.0):
    t_uniform = [float(round(x, 3)) for x in np.arange(N) * DT_NS]
    ch2_t = t_uniform
    if timebase is None:
        tb1 = {"unit": "ns", "delta": 1.0, "origin": 0.0}
    else:
        tb1 = timebase
    return {
        "CH1": {"samples": ch1_samples, "timebase": tb1,
                "vertical_gain": ch1_gain, "probe_attenuation": ch1_probe,
                "deskew": "0 ns"},
        "CH2": {"samples": ch2_samples,
                "timebase": {"unit": "ns", "delta": 1.0, "origin": 0.0},
                "vertical_gain": ch2_gain, "probe_attenuation": ch2_probe,
                "deskew": ch2_deskew},
        "trigger": {"index": trig_index, "timestamp": trig_time,
                    "unit": "ns"},
    }


def build_case() -> dict:
    t_tmpl = (np.arange(-12, 13) * 1.0).tolist()
    pulse = [float(round(_pulse(np.array([x]), 0.0)[0], 3)) for x in t_tmpl]

    # seg-02 uneven: event at true 60.4 ns; explicit per-sample t.
    s2_1, t2, tb2 = _uneven_axis(60.4)
    # CH2 on its own uniform axis, raw center 66.4 -> deskew -6 => 60.4
    s2_2 = _uniform_channel(66.4)
    # trigger true time on uneven axis:
    trig2 = float(round(t2[TRIG_I], 4))

    seg = []

    seg.append({
        "seg_id": "seg-01-reference", "order": 1,
        "label": "reference capture, uniform 1 ns",
        "reference_channel": "CH1",
        "channels": _ch(_uniform_channel(60.0), _uniform_channel(66.0)),
    })

    seg.append({
        "seg_id": "seg-02-uneven-timebase", "order": 2,
        "label": "explicit non-uniform dt pattern [1.3,1.0,0.6] ns",
        "reference_channel": "CH1",
        "channels": _ch(s2_1, s2_2, timebase=tb2,
                        trig_time=f"{trig2} ns"),
    })

    s3_1 = _uniform_channel(60.0 - 0.2, amp=34000.0, clip=True)
    s3_2 = _uniform_channel(65.8)
    seg.append({
        "seg_id": "seg-03-vertical-saturation", "order": 3,
        "label": "CH1 clips at +32767 inside window; CH2 probe label bad",
        "reference_channel": "CH1",
        "channels": _ch(s3_1, s3_2, ch2_probe="x10?",
                        trig_time="49.8 ns"),
        # trigger timestamp kept separately below; _ch default overridden:
    })
    seg[-1]["channels"]["trigger"]["timestamp"] = "49.8 ns"

    s4_1 = _uniform_channel(60.0, second=(78.0, 2000.0))
    s4_2 = _uniform_channel(66.0, second=(84.0, 2000.0))
    seg.append({
        "seg_id": "seg-04-multipeak-alignment", "order": 4,
        "label": "two correlation peaks (0 ns and +18 ns), both retained",
        "reference_channel": "CH1",
        "channels": _ch(s4_1, s4_2),
    })

    seg.append({
        "seg_id": "seg-05-channel-deskew", "order": 5,
        "label": "CH2 raw +6 ns corrected by -6 ns deskew",
        "reference_channel": "CH1",
        "channels": _ch(_uniform_channel(60.0), _uniform_channel(66.0)),
    })

    s6_1 = _uniform_channel(60.2, gaps=(58, 72))
    s6_2 = _uniform_channel(66.2, gaps=(58, 72))
    seg.append({
        "seg_id": "seg-06-missing-samples", "order": 6,
        "label": "gap (None) through most of the v1 measurement window",
        "reference_channel": "CH1",
        "channels": _ch(s6_1, s6_2, trig_time="50.2 ns"),
    })
    seg[-1]["channels"]["trigger"]["timestamp"] = "50.2 ns"

    seg.append({
        "seg_id": "seg-07-trigger-at-boundary", "order": 7,
        "label": "trigger mark sits on sample 0 (left edge)",
        "reference_channel": "CH1",
        "channels": _ch(_uniform_channel(60.0), _uniform_channel(66.0),
                        trig_index=0, trig_time="0 ns"),
    })

    s8_1 = _uniform_channel(60.1, second=(70.0, -600.0))
    s8_2 = _uniform_channel(66.1, second=(76.0, -600.0))
    seg.append({
        "seg_id": "seg-08-window-revision", "order": 8,
        "label": "secondary bump at 70 ns: v1 sees it, v2 does not",
        "reference_channel": "CH1",
        "channels": _ch(s8_1, s8_2, trig_time="50.1 ns"),
    })
    seg[-1]["channels"]["trigger"]["timestamp"] = "50.1 ns"

    windows = [
        {"version": "v1", "start": "48 ns", "end": "72 ns",
         "metric": "vpp", "note": "initial wide measurement window"},
        {"version": "v2", "start": "54 ns", "end": "66 ns",
         "metric": "vpp",
         "note": "tightened around primary pulse (parameter revision)"},
    ]

    return {
        "case_id": "wfline-fixed-2026-09",
        "title": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "coordinate_basis": {
            "time": "seconds (original strings carry ns)",
            "voltage": "volts at probe tip after gain x attenuation",
            "counts": "raw ADC codes; None = missing, +/-32767/32768 = rail",
        },
        "reference_pulse": {"unit": "ns", "t": t_tmpl, "samples": pulse},
        "windows": windows,
        "segments": seg,
        "replay": [
            {"step": 1, "action": "ingest",
             "expect_judgements": [
                 "time-uneven", "saturation-skipped", "multipeak-candidates",
                 "channel-deskew-applied", "gap-skipped",
                 "trigger-at-boundary", "bad-probe-attenuation"]},
            {"step": 2, "action": "revise-window", "from": "v1", "to": "v2"},
            {"step": 3, "action": "source-patch", "target_segment": "seg-08",
             "target_channel": "CH1",
             "effect": "only seg-08/CH1 derivations invalidate"},
            {"step": 4, "action": "conflict-decision",
             "segment": "seg-04",
             "note": "two reviewers accept different multi-peak candidates "
                     "from the same base; second submit sees 409 + diff"},
            {"step": 5, "action": "branch-ruling", "segment": "seg-04",
             "note": "adopt candidate 1; candidate 2 stays not-adopted"},
            {"step": 6, "action": "undo-to-branch",
             "note": "fork before the ruling; pre-ruling ambiguity restored"},
        ],
    }


def write_case(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_case(), ensure_ascii=False, indent=1))
    return p


if __name__ == "__main__":
    out = write_case(pathlib.Path(__file__).resolve().parents[1]
                     / "cases" / "waveform-lineage" / "case.json")
    print(out)
