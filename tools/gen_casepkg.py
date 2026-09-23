"""生成案例包 casepkg/wave-overlay-demo（波形叠影台：示波器分段、触发抖动与测量窗口谱系）。

每个边界情形对应真实段数据：
  ch1/0002 不等间隔时基 | ch1/0003 竖直饱和 | ch1/0004 多峰对齐
  ch2/0005 通道 deskew | ch1/0006 缺样 | ch1/0007 触发在边界
  ch1/0008 单位无法转换（保留原位+诊断） | intervals v1->v2 区间版本修订
"""
import json
import os

import numpy as np

PKG = os.path.join(os.path.dirname(__file__), "..", "casepkg", "wave-overlay-demo")
CASE = "wave-overlay-demo"
N = 512
DT = 200e-12          # 200 ps
AMP = 60e-3           # 60 mV
WIDTH = 2.5e-9
FULL_SCALE = 80e-3    # 80 mV


def pulse(t, te, amp=AMP):
    return amp * np.exp(-((t - te) / WIDTH) ** 2)


def base_segment(seg_id, channel, te, trigger_t, deskew_s=0.0, timebase=None,
                 samples=None, sample_unit="mV", extra=None):
    tb = timebase or {"kind": "uniform",
                      "t0": {"value": 0.0, "unit": "s"},
                      "dt": {"value": 200.0, "unit": "ps"},
                      "n": N}
    seg = {
        "id": seg_id, "case": CASE, "channel": channel,
        "timebase": tb,
        "samples": [round(float(x), 6) for x in samples],
        "sample_unit": sample_unit,
        "vertical_gain": {"value": 10.0, "unit": "mV/div"},
        "probe_attenuation": {"value": 10.0, "unit": "x"},
        "full_scale": {"value": 80.0, "unit": "mV"},
        "trigger": {"timestamp": {"value": trigger_t * 1e9, "unit": "ns"},
                    "marker": f"trig:{seg_id}"},
        "deskew": {"value": deskew_s * 1e12, "unit": "ps"},
    }
    if extra:
        seg.update(extra)
    return seg


def main():
    os.makedirs(os.path.join(PKG, "segments"), exist_ok=True)
    t = DT * np.arange(N)
    segs = {}

    # 参考段
    segs["ch1_0000"] = base_segment("seg:ch1/0000", "ch1", 50e-9, 50.1e-9,
                                    samples=pulse(t, 50e-9) * 1e3)
    # 常规抖动 +1.2 ns
    segs["ch1_0001"] = base_segment("seg:ch1/0001", "ch1", 51.2e-9, 51.25e-9,
                                    samples=pulse(t, 51.2e-9) * 1e3)
    # 不等间隔时基（真实时间轴非均匀，禁止索引相减）
    dt_var = DT * (1.0 + 0.04 * np.sin(np.arange(N) * 0.7))
    t_non = np.concatenate([[0.0], np.cumsum(dt_var)[:-1]])
    segs["ch1_0002"] = base_segment(
        "seg:ch1/0002", "ch1", 49.2e-9, 49.25e-9,
        timebase={"kind": "nonuniform",
                  "t": [round(float(x), 15) for x in t_non], "t_unit": "s"},
        samples=pulse(t_non, 49.2e-9) * 1e3)
    # 竖直饱和（150 mV 信号被 80 mV 满量程截顶）
    sat = np.clip(pulse(t, 50.4e-9, amp=150e-3), 0, FULL_SCALE)
    segs["ch1_0003"] = base_segment("seg:ch1/0003", "ch1", 50.4e-9, 50.5e-9,
                                    samples=sat * 1e3)
    # 多峰对齐（双脉冲 -> 互相关双峰）
    two = pulse(t, 42e-9, amp=40e-3) + pulse(t, 58e-9, amp=40e-3)
    segs["ch1_0004"] = base_segment("seg:ch1/0004", "ch1", 42e-9, 42.1e-9,
                                    samples=two * 1e3)
    # 通道 deskew（ch2 标定 350 ps）
    segs["ch2_0005"] = base_segment("seg:ch2/0005", "ch2", 50.5e-9, 50.6e-9,
                                    deskew_s=350e-12,
                                    samples=pulse(t, 50.5e-9) * 1e3)
    # 缺样（索引 300..319 丢失）
    segs["ch1_0006"] = base_segment("seg:ch1/0006", "ch1", 50.8e-9, 50.9e-9,
                                    samples=pulse(t, 50.8e-9) * 1e3,
                                    extra={"missing": [[300, 20]]})
    # 触发在边界（触发时标落在段首）
    segs["ch1_0007"] = base_segment("seg:ch1/0007", "ch1", 50e-9, 0.0,
                                    samples=pulse(t, 50e-9) * 1e3)
    # 单位无法转换（保留原值 + 诊断，不参与派生）
    segs["ch1_0008"] = base_segment("seg:ch1/0008", "ch1", 50e-9, 50.1e-9,
                                    samples=pulse(t, 50e-9) / 10e-3,
                                    sample_unit="divs")
    # 源数据局部改动：ch2 deskew 标定修订 350ps -> 300ps（只影响 ch2/0005）
    segs["ch2_0005_rev2"] = base_segment("seg:ch2/0005", "ch2", 50.5e-9, 50.6e-9,
                                         deskew_s=300e-12,
                                         samples=pulse(t, 50.5e-9) * 1e3)

    for name, seg in segs.items():
        with open(os.path.join(PKG, "segments", f"{name}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(seg, fh, ensure_ascii=False, indent=1)

    intervals_v1 = {"version": "intervals:v1", "case": CASE, "windows": [
        {"id": "win:rise", "t0": {"value": -4.0, "unit": "ns"},
         "t1": {"value": 0.0, "unit": "ns"}},
        {"id": "win:top", "t0": {"value": 0.0, "unit": "ns"},
         "t1": {"value": 4.0, "unit": "ns"}}]}
    intervals_v2 = {"version": "intervals:v2", "case": CASE, "windows": [
        {"id": "win:rise", "t0": {"value": -3.0, "unit": "ns"},
         "t1": {"value": 0.0, "unit": "ns"}},
        {"id": "win:top", "t0": {"value": 0.0, "unit": "ns"},
         "t1": {"value": 6.0, "unit": "ns"}},
        {"id": "win:tail", "t0": {"value": 6.0, "unit": "ns"},
         "t1": {"value": 12.0, "unit": "ns"}}]}
    for name, obj in (("intervals_v1", intervals_v1), ("intervals_v2", intervals_v2)):
        with open(os.path.join(PKG, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)

    manifest = {
        "case_id": CASE,
        "title": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "story": "信号完整性团队导入分段捕获、触发标记和通道标定，对齐重复事件并审查测量分布。",
        "sources": (
            [{"kind": "segment", "name": f"segments/{n}", "file": f"segments/{n}.json"}
             for n in ["ch1_0000", "ch1_0001", "ch1_0002", "ch1_0003", "ch1_0004",
                       "ch2_0005", "ch1_0006", "ch1_0007", "ch1_0008"]]
            + [{"kind": "segment-patch", "name": "segments/ch2_0005_rev2",
                "file": "segments/ch2_0005_rev2.json"}]
            + [{"kind": "intervals", "name": "intervals:v1", "file": "intervals_v1.json"},
               {"kind": "intervals", "name": "intervals:v2", "file": "intervals_v2.json"}]),
    }
    with open(os.path.join(PKG, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)

    scenario = {
        "case_id": CASE,
        "steps": [
            {"id": "s1", "kind": "import-baseline",
             "title": "初始导入：每段原采样/时基/增益/衰减/触发/deskew/区间版本化"},
            {"id": "s2", "kind": "review-baseline",
             "title": "审阅者核对基线",
             "note": "基线核对完成：9 段 + ch2 修订源 + 区间 v1/v2 均在案"},
            {"id": "s3", "kind": "param-revision", "title": "参数修订：峰阈值 0.8 -> 0.9",
             "params_version": "params:v2"},
            {"id": "s4", "kind": "interval-revision", "title": "区间版本修订 v1 -> v2",
             "interval_version": "intervals:v2"},
            {"id": "s5", "kind": "source-patch",
             "title": "源数据局部改动：ch2 deskew 350ps -> 300ps",
             "source_id": f"src:{CASE}/segments/ch2_0005",
             "new_source_id": f"src:{CASE}/segments/ch2_0005_rev2"},
            {"id": "s6", "kind": "conflicting-decisions",
             "title": "互相冲突的人工决定：多峰段 ch1/0004 选峰",
             "decisions": [
                 {"target": "cand:seg:ch1/0004/0", "decision": "accept",
                  "note": "审阅A：取 τ=+8ns 峰"},
                 {"target": "cand:seg:ch1/0004/1", "decision": "accept",
                  "note": "审阅B：取 τ=-8ns 峰"}]},
            {"id": "s7", "kind": "edge-branch",
             "title": "第二分支：检验 R1 不等间隔/R2 缺样饱和/R3 多峰候选",
             "note": "保留全部合法候选，不收敛到单峰"},
        ],
    }
    with open(os.path.join(PKG, "scenario.json"), "w", encoding="utf-8") as fh:
        json.dump(scenario, fh, ensure_ascii=False, indent=1)

    counterexamples = {
        "case_id": CASE,
        "archive": "反例档案：每次只改动一个直接来源，影响必须被限制在正确范围",
        "cases": [
            {"id": "ce:nonuniform-timebase", "segment": "seg:ch1/0002",
             "expect_rule": "nonuniform-no-index-sub",
             "basis": "时间轴不等间隔时不用索引相减"},
            {"id": "ce:vertical-saturation", "segment": "seg:ch1/0003",
             "expect_rule": "gap-no-interp",
             "basis": "饱和区间不进入插值"},
            {"id": "ce:multipeak-alignment", "segment": "seg:ch1/0004",
             "expect_rule": "multipeak-candidates",
             "basis": "对齐保留多峰互相关候选"},
            {"id": "ce:channel-deskew", "segment": "seg:ch2/0005",
             "expect_rule": None,
             "basis": "deskew 改动只使本通道派生失效"},
            {"id": "ce:missing-samples", "segment": "seg:ch1/0006",
             "expect_rule": "gap-no-interp",
             "basis": "缺样区间不进入插值"},
            {"id": "ce:trigger-at-boundary", "segment": "seg:ch1/0007",
             "expect_rule": "trigger-at-boundary",
             "basis": "触发在边界的段被排除并记录原因"},
            {"id": "ce:interval-revision", "segment": None,
             "expect_rule": None,
             "basis": "区间版本修订只使测量派生失效，不影响对齐"},
        ],
    }
    with open(os.path.join(PKG, "counterexamples.json"), "w", encoding="utf-8") as fh:
        json.dump(counterexamples, fh, ensure_ascii=False, indent=1)
    print("casepkg generated:", PKG)


if __name__ == "__main__":
    main()
