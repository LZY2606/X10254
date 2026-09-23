"""离线案例包：波形叠影台——示波器分段、触发抖动与测量窗口谱系。

叙事：信号完整性团队导入分段捕获、触发标记和通道标定，对齐重复事件
并审查测量分布。案例固定包含七种边界情形：
  seg-02 缺样；seg-03 不等间隔时基；seg-04 竖直饱和；seg-05 触发在边界；
  seg-07 通道 deskew（ch2 = 25 ps，段级 deskew 单位不可转换 -> 诊断）；
  全部段 多峰对齐（双沿重复事件）；mi-v1 -> mi-v2 区间版本修订。
"""
from __future__ import annotations

import json
import os

import numpy as np

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "casepack")

N_SAMPLES = 64
DT_S = 0.5e-9
T0_S = 500e-9
GAIN = 0.01            # V/adc_code
SAT_CODE = 1023
EDGE1_S = 8e-9
EDGE2_S = 24e-9
CH2_DESKEW_S = 25e-12

# 每段触发抖动（秒），seg-05 触发在边界单独处理
JITTER = {
    "seg-01": 0.0e-9, "seg-02": 0.3e-9, "seg-03": -0.2e-9, "seg-04": 0.1e-9,
    "seg-05": 0.4e-9, "seg-06": -0.3e-9, "seg-07": 0.2e-9,
}
MISSING_SEG02 = [20, 21, 22, 23]


def _pulse(t, center, width=0.8e-9):
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def _signal(t, shift_s):
    return 0.5 * _pulse(t, EDGE1_S + shift_s) + 0.5 * _pulse(t, EDGE2_S + shift_s)


def _codes(rng, t, shift_s, scale=1.0, gain=GAIN, clip=None):
    volts = _signal(t, shift_s) * scale
    codes = volts / gain + rng.normal(0.0, 0.2, size=len(t))
    codes = np.round(codes)
    if clip is not None:
        codes = np.minimum(codes, clip)
    return [int(c) for c in codes]


def _quantity(value, unit, basis=None):
    q = {"value": value, "unit": unit}
    if basis:
        q["basis"] = basis
    return q


def build_baseline():
    rng = np.random.default_rng(1225)
    t_uniform = T0_S + np.arange(N_SAMPLES) * DT_S

    # seg-03：不等间隔时基（显式时间轴，dt 在 0.46/0.54 ns 间交替）
    dts = np.array([DT_S * (0.92 if i % 2 == 0 else 1.08) for i in range(N_SAMPLES - 1)])
    t_nonuniform = T0_S + np.concatenate([[0.0], np.cumsum(dts)])

    segments = []
    for seg_id in sorted(JITTER):
        shift = JITTER[seg_id]
        channel = "ch2" if seg_id == "seg-07" else "ch1"
        if seg_id == "seg-03":
            timebase = {"kind": "explicit",
                        "t": [float(x) for x in t_nonuniform], "unit": "s"}
            t = t_nonuniform
        else:
            timebase = {"kind": "uniform",
                        "t0": _quantity(T0_S, "s"), "dt": _quantity(DT_S, "s")}
            t = t_uniform
        scale = 30.0 if seg_id == "seg-04" else 1.0
        values = _codes(rng, t - T0_S, shift, scale=scale, clip=SAT_CODE)
        missing = []
        if seg_id == "seg-02":
            missing = list(MISSING_SEG02)
            for i in missing:
                values[i] = None
        if seg_id == "seg-05":
            trigger = _quantity(T0_S, "s", "acq_clock")  # 触发在边界
        else:
            trigger = _quantity(T0_S + EDGE1_S + shift, "s", "acq_clock")
        deskew = (_quantity(5, "tick", "dac") if seg_id == "seg-07"
                  else _quantity(0.0, "s"))
        segments.append({
            "segment_id": seg_id,
            "channel": channel,
            "samples": {"values": values, "unit": "adc_code", "basis": "adc:10bit"},
            "timebase": timebase,
            "vertical_gain": _quantity(GAIN, "V/adc_code"),
            "probe_attenuation": _quantity(10.0, "ratio"),
            "trigger_time": trigger,
            "deskew": deskew,
            "missing": missing,
            "saturation_code": SAT_CODE,
            "interval_version": "mi-v1",
        })

    return {
        "case_id": "waveform-overlay-desk-001",
        "title": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "narrative": "信号完整性团队导入分段捕获、触发标记和通道标定，"
                     "对齐重复事件并审查测量分布。",
        "channels": [
            {"channel_id": "ch1", "deskew": _quantity(0.0, "s")},
            {"channel_id": "ch2", "deskew": _quantity(CH2_DESKEW_S, "s")},
        ],
        "intervals": {
            "versions": [
                {"version": "mi-v1",
                 "start": _quantity(6e-9, "s", "aligned"),
                 "end": _quantity(10e-9, "s", "aligned")},
                {"version": "mi-v2",
                 "start": _quantity(7e-9, "s", "aligned"),
                 "end": _quantity(12e-9, "s", "aligned")},
            ],
            "active": "mi-v1",
        },
        "segments": segments,
    }


def build_revisions(baseline):
    """一次参数修订、一次只影响局部依赖的源数据改动、一次互相冲突的人工决定。"""
    rng = np.random.default_rng(772)
    seg04_old = next(s for s in baseline["segments"] if s["segment_id"] == "seg-04")
    t = T0_S + np.arange(N_SAMPLES) * DT_S
    fixed = dict(seg04_old)
    fixed["vertical_gain"] = _quantity(0.002, "V/adc_code")
    fixed["samples"] = {"values": _codes(rng, t - T0_S, JITTER["seg-04"], gain=0.002),
                        "unit": "adc_code", "basis": "adc:10bit"}
    return {
        "param_revision": {
            "revision_id": "rev-param-mi-v2",
            "kind": "param",
            "param": "measurement_interval",
            "from": "mi-v1",
            "to": "mi-v2",
            "author": "si-reviewer",
            "note": "区间版本修订：测量窗由 6~10ns 改为 7~12ns（aligned 基准）",
        },
        "source_changes": [{
            "change_id": "rev-src-seg-04",
            "kind": "source",
            "source_id": "seg-04",
            "version": 2,
            "author": "si-reviewer",
            "note": "竖直增益误录修正（30x -> 1x），只应影响 seg-04 的派生",
            "payload": fixed,
        }],
        "conflicting_decisions": [
            {"decision_id": "dec-A", "segment_id": "seg-01",
             "candidate_id": "cand-seg-01-1", "decision": "accept",
             "reviewer": "reviewer-A",
             "note": "主瓣候选，相关得分最高"},
            {"decision_id": "dec-B", "segment_id": "seg-01",
             "candidate_id": "cand-seg-01-2", "decision": "accept",
             "reviewer": "reviewer-B",
             "note": "次峰候选，与 dec-A 冲突，双方差异需可见"},
        ],
    }


def build_counterpoints():
    """反例档案：每次只改动一个直接来源，检验影响被限制在正确范围。

    被采纳与未采纳的结果都保留；判断依据写成机器可读取的事件
    （rationale 列表，rule 引用 core.RULES 的规则 id）。
    """
    meas_all = ["meas-seg-%02d" % i for i in range(1, 8)]
    return [
        {"cp_id": "cp-nonuniform-timebase", "condition": "不等间隔时基",
         "segment": "seg-03",
         "rule": "nonuniform-timebase-no-index-subtraction",
         "expected_affected": ["align-seg-03", "jitter-seg-03",
                               "meas-seg-03", "overlay-seg-03"],
         "accepted": [{"id": "jitter-seg-03", "basis": "actual-time-axis"}],
         "rejected": [{"id": "jitter-seg-03-naive-index",
                       "basis": "index-subtraction",
                       "note": "索引相减结果被否定，保留备查"}],
         "rationale": [{"event": "rule-applied",
                        "rule": "nonuniform-timebase-no-index-subtraction",
                        "detail": "seg-03 时基不等间隔，时间差由实际时间轴逐点相减"}]},
        {"cp_id": "cp-vertical-saturation", "condition": "竖直饱和",
         "segment": "seg-04",
         "rule": "gap-saturation-excluded-from-interpolation",
         "expected_excluded_reason": "vertical-saturation",
         "expected_affected": ["align-seg-04", "jitter-seg-04",
                               "meas-seg-04", "overlay-seg-04"],
         "accepted": [{"id": "meas-seg-04", "status": "excluded"}],
         "rejected": [{"id": "meas-seg-04-clipped-interp",
                       "note": "跨饱和区插值的结果被否定，保留备查"}],
         "rationale": [{"event": "rule-applied",
                        "rule": "gap-saturation-excluded-from-interpolation",
                        "detail": "seg-04 饱和区间不进入插值"}]},
        {"cp_id": "cp-multi-peak-alignment", "condition": "多峰对齐",
         "segment": "seg-01",
         "rule": "multi-peak-alignment-candidates-retained",
         "min_candidates": 2,
         "expected_affected": ["align-seg-01"],
         "accepted": [{"id": "cand-seg-01-1", "decision": "accept"}],
         "rejected": [{"id": "cand-seg-01-2", "decision": "reject",
                       "note": "次峰候选被否定但保留，可随时改判"}],
         "rationale": [{"event": "rule-applied",
                        "rule": "multi-peak-alignment-candidates-retained",
                        "detail": "双沿重复事件互相关存在多峰，全部合法候选保留"}]},
        {"cp_id": "cp-channel-deskew", "condition": "通道 deskew",
         "segment": "seg-07",
         "rule": None,
         "expected_deskew_s": CH2_DESKEW_S,
         "expected_affected": ["jitter-seg-07", "overlay-seg-07"],
         "accepted": [{"id": "jitter-seg-07", "deskew_applied_s": CH2_DESKEW_S}],
         "rejected": [{"id": "jitter-seg-07-no-deskew",
                       "note": "未应用通道 deskew 的结果被否定，保留备查"}],
         "rationale": [{"event": "calibration-applied",
                        "rule": None,
                        "detail": "ch2 通道 deskew 25ps 进入抖动计算；"
                                  "段级 deskew 单位 tick 不可转换，留原位并附诊断"}]},
        {"cp_id": "cp-missing-samples", "condition": "缺样",
         "segment": "seg-02",
         "rule": "gap-saturation-excluded-from-interpolation",
         "expected_excluded_reason": "missing-samples",
         "expected_affected": ["align-seg-02", "jitter-seg-02",
                               "meas-seg-02", "overlay-seg-02"],
         "accepted": [{"id": "meas-seg-02", "status": "excluded"}],
         "rejected": [{"id": "meas-seg-02-gap-interp",
                       "note": "跨缺样区插值的结果被否定，保留备查"}],
         "rationale": [{"event": "rule-applied",
                        "rule": "gap-saturation-excluded-from-interpolation",
                        "detail": "seg-02 缺样区间不进入插值"}]},
        {"cp_id": "cp-trigger-at-boundary", "condition": "触发在边界",
         "segment": "seg-05",
         "rule": None,
         "expected_excluded_reason": "trigger-at-boundary",
         "expected_affected": ["jitter-seg-05", "meas-seg-05", "overlay-seg-05"],
         "accepted": [{"id": "overlay-seg-05", "status": "excluded"}],
         "rejected": [{"id": "jitter-seg-05-boundary-mean",
                       "note": "把边界触发纳入抖动均值的结果被否定，保留备查"}],
         "rationale": [{"event": "segment-excluded",
                        "rule": None,
                        "detail": "触发时标落在采集边界，段被排除但保留显示"}]},
        {"cp_id": "cp-interval-version-revision", "condition": "区间版本修订",
         "segment": None,
         "rule": None,
         "revision": "rev-param-mi-v2",
         "expected_affected": meas_all,
         "expected_unaffected_prefix": "jitter-",
         "accepted": [{"id": "meas-seg-06", "interval_version": "mi-v2"}],
         "rejected": [{"id": "global-silent-recompute",
                       "note": "全局悄悄替换旧结论的做法被否定；"
                               "旧结论按事件保留，可恢复"}],
         "rationale": [{"event": "invalidation-scoped",
                        "rule": None,
                        "detail": "mi-v1 -> mi-v2 只使依赖 mi-v1 的测量失效，"
                                  "抖动等不受影响"}]},
    ]


def baseline_sources(data):
    """把基线案例拆成来源记录（source_id, version, kind, payload）。"""
    items = [
        ("case-meta", 1, "case-meta",
         {"case_id": data["case_id"], "title": data["title"],
          "narrative": data["narrative"]}),
        ("intervals", 1, "interval-set", data["intervals"]),
    ]
    for ch in data["channels"]:
        items.append(("cal-%s" % ch["channel_id"], 1, "channel", ch))
    for seg in data["segments"]:
        items.append((seg["segment_id"], 1, "segment", seg))
    return items


def _load(name):
    with open(os.path.join(BASE_DIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_baseline():
    return _load("baseline.json")


def load_revisions():
    return _load("revisions.json")


def load_counterpoints():
    return _load("counterpoints.json")
