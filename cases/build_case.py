"""Build the domain-specific offline case pack:
'波形叠影台：示波器分段、触发抖动与测量窗口谱系'.

The committed output (cases/waveform-overlay-lineage/) carries, for every
segment, its raw samples, timebase, vertical gain, probe attenuation, trigger
mark, channel deskew and versioned measurement windows -- all in original
representations with units. Normalization is a derived-layer concern.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "waveform-overlay-lineage")

ADC_FULL = 2048
GAIN = "5 mV/code"
N = 201
T0, DT = "0 ns", "1 ns"


def bump(t_ns, center_ns, width_ns=8.0, amp_codes=200):
    """A pulse in physical time. 1 code = 5 mV via the channel calibration."""
    t_ns = np.asarray(t_ns, dtype=float)
    return amp_codes * np.exp(
        -0.5 * ((t_ns - center_ns) / (width_ns / 2.355)) ** 2)


def clip(values):
    values = np.rint(np.asarray(values, dtype=float)).astype(int)
    return np.clip(values, -ADC_FULL, ADC_FULL).tolist()


def seg(seg_id, channel, trigger, samples, timebase=None,
        gain=GAIN, atten="10x", deskew="0 ns", note=""):
    return {
        "segment_id": seg_id,
        "channel_id": channel,
        "note": note,
        "raw": {
            "timebase": timebase or {"kind": "uniform", "t0": T0, "dt": DT,
                                     "n": len(samples)},
            "samples": samples,
            "adc_full_scale": ADC_FULL,
            "vertical_gain": gain,
            "probe_attenuation": atten,
            "trigger_mark": trigger,
        },
        "channel_deskew": deskew,
    }


def build_segments():
    uniform = np.arange(N, dtype=float)
    segments = []

    # S1 reference: clean pulse, trigger on the 20 ns mark.
    segments.append(seg("S1", "CH1", "20 ns", clip(bump(uniform, 20)),
                        note="reference segment"))

    # S2 repeated event arriving 5 ns early vs its trigger -> 5 ns jitter.
    segments.append(seg("S2", "CH1", "20 ns", clip(bump(uniform, 15)),
                        note="repeated event with 5 ns lead"))

    # S3 irregular, monotonic timebase (1 ns then 2 ns through a stretch).
    times_ns = []
    t = 0.0
    while len(times_ns) < N:
        times_ns.append(t)
        t += 2.0 if 40 <= len(times_ns) < 120 else 1.0
    times_ns = times_ns[:N]
    times_raw = [f"{x:g} ns" for x in times_ns]
    segments.append(seg(
        "S3", "CH1", "20 ns", clip(bump(np.array(times_ns), 20)),
        timebase={"kind": "list", "times": times_raw},
        note="non-uniform timebase: physical timestamps required"))

    # S4 vertical saturation: pulse driven into the ADC rails.
    segments.append(seg("S4", "CH1", "20 ns",
                        clip(bump(uniform, 20, amp_codes=2600)),
                        note="vertical saturation through the pulse"))

    # S5 multi-peak alignment: two identical pulses (20 ns and 220 ns).
    s5 = clip(bump(uniform, 20) + bump(uniform, 220))
    segments.append(seg("S5", "CH1", "20 ns", s5,
                        note="double pulse: two legal alignment candidates"))

    # S6 missing samples around the pulse (None; never filled by interpolation).
    s6 = clip(bump(uniform, 20))
    for i in range(15, 26):
        s6[i] = None
    segments.append(seg("S6", "CH1", "20 ns", s6,
                        note="missing samples inside the acquisition"))

    # S7 trigger mark exactly on the record boundary.
    segments.append(seg("S7", "CH1", "0 ns", clip(bump(uniform, 20)),
                        note="trigger mark on record boundary"))

    # S8 ordinary late repeat event: +3 ns lag.
    segments.append(seg("S8", "CH1", "20 ns", clip(bump(uniform, 23)),
                        note="late event, +3 ns"))

    # S9 CH2 probe with 5 ns deskew; pulse 6 ns late, trigger 20 ns.
    segments.append(seg("S9", "CH2", "20 ns", clip(bump(uniform, 26)),
                        atten="1x", deskew="5 ns",
                        note="CH2 probe, 5 ns deskew"))

    # S10 un-parseable probe attenuation: raw retained, not converted.
    segments.append(seg("S10", "CH1", "20 ns", clip(bump(uniform, 20)),
                        atten="ten x?",
                        note="unconvertible attenuation: raw + diagnostic"))

    return segments


def build_case():
    segments = build_segments()
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {
        "case_id": "case:waveform-overlay-lineage",
        "title": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "domain": "signal-integrity/oscilloscope-segmented-capture",
        "adc_full_scale": ADC_FULL,
        "channels": [
            {"channel_id": "CH1", "vertical_gain": "5 mV/code",
             "probe_attenuation": "10x", "deskew": "0 ns",
             "label": "CLK probe A"},
            {"channel_id": "CH2", "vertical_gain": "5 mV/code",
             "probe_attenuation": "1x", "deskew": "5 ns",
             "label": "DATA probe B"},
        ],
        "segments": [s["segment_id"] for s in segments],
        "windows": ["W1", "W2"],
        "params": {"peak_threshold": 0.3, "tie_tolerance": 0.005,
                   "rule_version": "rules-2026.09"},
        "narrative": [
            "导入分段捕获、触发标记与通道标定，对齐重复事件并审查测量分布。",
            "初始核对：每段原采样、时基、竖直增益、探头衰减、触发时标、"
            "deskew、测量窗口版本化。",
            "边界组合：不等间隔时基、竖直饱和、多峰对齐、通道 deskew、缺样、"
            "触发在边界、区间版本修订。",
        ],
    }
    with open(os.path.join(OUT_DIR, "case.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    for segment in segments:
        with open(os.path.join(OUT_DIR, f"{segment['segment_id']}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(segment, f, ensure_ascii=False, indent=2)
    windows = [
        {"window_id": "W1", "label": "post-trigger eye",
         "t0": "10 ns", "t1": "60 ns", "note": "v1 initial window"},
        {"window_id": "W2", "label": "long hold",
         "t0": "120 ns", "t1": "180 ns",
         "note": "segments with no valid samples here are excluded"},
    ]
    with open(os.path.join(OUT_DIR, "windows.json"), "w", encoding="utf-8") as f:
        json.dump({"windows": windows}, f, ensure_ascii=False, indent=2)
    return manifest, segments, windows


if __name__ == "__main__":
    manifest, segments, windows = build_case()
    print(f"wrote {len(segments)} segments to {OUT_DIR}")
