"""Generates casepack/case.json — the offline domain case package.

"波形叠影台：示波器分段、触发抖动与测量窗口谱系" — a signal-integrity team
imports segmented captures, trigger markers and channel calibration, aligns
repeated events and reviews the measurement distribution.

The initial version fully presents, per segment: raw samples, timebase,
vertical gain, probe attenuation, trigger timestamp, deskew and the
measurement-window version. The package then schedules one parameter
revision, one source-data change with local dependency scope, and one pair
of mutually conflicting human decisions (replayed through the API).
"""
import json
import math
import os

DT = 0.2          # ns, nominal sample interval
N = 64            # samples per segment
T0 = 10.0         # ns, nominal event position
WIDTH = 0.5       # ns, pulse width


def pulse(t, t0=T0, amp=0.8, width=WIDTH):
    return amp * math.exp(-((t - t0) / width) ** 2)


def seg(key, channel, t_vals, v_actual, trigger_ns, deskew_ns,
        missing=(), saturated=(), v_unit="mV", clip_v=None):
    """v_actual in volts at the DUT; scope sees v/10 (x10 probe), stores mV."""
    v_scope_mv = []
    sat = set(saturated)
    for i, vv in enumerate(v_actual):
        raw = vv / 10.0 * 1000.0
        if clip_v is not None and raw >= clip_v / 10.0 * 1000.0:
            raw = clip_v / 10.0 * 1000.0
            sat.add(i)
        v_scope_mv.append(round(raw, 4))
    return {
        "seg_key": key,
        "channel": channel,
        "t": {"unit": "ns", "kind": "explicit", "values": [round(x, 4) for x in t_vals]},
        "v": {"unit": v_unit, "values": v_scope_mv},
        "vertical_gain": {"value": 100, "unit": "mV/div"},
        "probe_atten": {"value": 10, "unit": "x"},
        "trigger_ts": {"value": trigger_ns, "unit": "ns"},
        "deskew": {"value": deskew_ns, "unit": "ns"},
        "meas_window_version": "mw-v1",
        "flags": {"missing": sorted(missing), "saturated": sorted(sat)},
    }


def uniform_t():
    return [i * DT for i in range(N)]


def nonuniform_t():
    # deterministic wobble, monotonic, NOT index * dt
    out = []
    t = 0.0
    for i in range(N):
        out.append(t)
        t += DT + 0.04 * math.sin(i * 1.7) + (0.02 if i % 5 == 0 else -0.01)
    return out


def build():
    segs = []

    # seg-001 ch1 reference, clean uniform capture
    t = uniform_t()
    segs.append(seg("seg-001", "ch1", t, [pulse(x) for x in t], 10.01, 0.0))

    # seg-002 ch1 non-uniform timebase — index subtraction would be wrong
    t = nonuniform_t()
    segs.append(seg("seg-002", "ch1", t, [pulse(x, t0=10.02) for x in t], 10.04, 0.0))

    # seg-003 ch1 vertical saturation (clipped at 0.6 V equivalent)
    t = uniform_t()
    segs.append(seg("seg-003", "ch1", t, [pulse(x, t0=9.98, amp=0.9) for x in t],
                    9.96, 0.0, clip_v=0.6))

    # seg-004 ch2 deskew +0.30 ns (later patched to 0.25 by the scenario)
    t = uniform_t()
    segs.append(seg("seg-004", "ch2", t, [pulse(x, t0=10.05) for x in t], 10.08, 0.30))

    # seg-005 ch2 missing samples near the pulse edge
    t = uniform_t()
    segs.append(seg("seg-005", "ch2", t, [pulse(x, t0=10.03) for x in t],
                    10.02, 0.30, missing=(44, 45, 46, 47, 48)))

    # seg-006 ch3 trigger exactly on the window start boundary (10.0 ns)
    t = uniform_t()
    segs.append(seg("seg-006", "ch3", t, [pulse(x, t0=10.06) for x in t], 10.00, -0.20))

    # seg-007 ch1 double pulse -> ambiguous alignment, two correlation peaks
    t = uniform_t()
    segs.append(seg("seg-007", "ch1", t,
                    [pulse(x, t0=10.0, amp=0.8) + pulse(x, t0=11.2, amp=0.7) for x in t],
                    9.94, 0.0))

    # seg-008 ch3 voltage recorded in scope divisions — unconvertible here,
    # must stay raw with a diagnostic instead of being dropped for plotting
    t = uniform_t()
    s8 = seg("seg-008", "ch3", t, [pulse(x, t0=10.04) for x in t], 10.01, -0.20,
             v_unit="div")
    s8["v"]["values"] = [round(x / 10.0, 4) for x in s8["v"]["values"]]
    segs.append(s8)

    case = {
        "format": "waveform-overlay-case/1",
        "name": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "narrative": (
            "信号完整性团队导入分段捕获、触发标记和通道标定，对齐重复事件并审查测量分布。"
            "初始版本完整呈现每段原采样、时基、竖直增益、探头衰减、触发时标、deskew 和测量区间版本化；"
            "随后安排一次参数修订、一次只影响局部依赖的源数据改动和一次互相冲突的人工决定。"
        ),
        "params": {
            "align_threshold": 0.6,
            "expected_trigger_ns": 10.0,
            "jitter_bin_ns": 0.02,
        },
        "meas_windows": {
            "mw-v1": {"start_ns": 10.0, "end_ns": 20.0,
                      "start_inclusive": False, "end_inclusive": True},
            "mw-v2": {"start_ns": 10.0, "end_ns": 22.0,
                      "start_inclusive": True, "end_inclusive": True},
        },
        "initial_mw_version": "mw-v1",
        "segments": segs,
        "scenario": [
            {"step": "param_revision", "key": "align_threshold",
             "from": 0.6, "to": 0.5,
             "note": "参数修订：降低对齐候选阈值"},
            {"step": "source_patch", "segment": "seg-004",
             "patch": {"deskew": {"value": 0.25, "unit": "ns"}},
             "note": "源数据改动：仅 seg-004 的 deskew 标定修正，失效范围必须限于其依赖"},
            {"step": "window_revision", "from": "mw-v1", "to": "mw-v2",
             "note": "测量区间版本修订：边界改为包含，末端扩展到 22 ns"},
            {"step": "conflict", "candidate": "cand:seg-007#1",
             "note": "两个复核窗口对同一多峰候选做出互相冲突的裁定"},
        ],
    }
    return case


if __name__ == "__main__":
    case = build()
    path = os.path.join(os.path.dirname(__file__), "casepack", "case.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case, f, ensure_ascii=False, indent=1)
    print("wrote", path, "segments:", len(case["segments"]))
