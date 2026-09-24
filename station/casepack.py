"""仓库内离线案例包生成：波形叠影台固定领域材料。

固定叙事包含：不等间隔时基、竖直饱和、多峰对齐、通道 deskew、缺样、
触发在边界、区间版本修订。所有领域量带单位/坐标基准与原始表示。
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional

CASE_ID = "oscilloscope-lineage-v1"
ADC_BASE = 128
ADC_FS = 255


def gaussian(grid: List[Optional[float]], centers, amp, sigma=1.6, clip=True):
    out = [ADC_BASE] * len(grid)
    for c in centers:
        for i, t in enumerate(grid):
            if t is None:
                continue
            v = ADC_BASE + amp * math.exp(-0.5 * ((t - c) / sigma) ** 2)
            if clip:
                v = min(v, ADC_FS)
            if v > out[i]:
                out[i] = v
    return out


def q(raw, unit, basis):
    return {"raw_repr": raw, "unit": unit, "coordinate_basis": basis}


def regular_grid(n, dt=1.0, start=0.0):
    return [round(start + i * dt, 6) for i in range(n)]


def irregular_grid():
    # 0..20ns 等间隔 1ns，之后改为 2ns：索引相减会给出错误时间。
    return [float(i) for i in range(21)] + [22.0 + 2 * i for i in range(1, 12)]


def samples_payload(grid, values, missing=(), note=""):
    vals = [None if i in set(missing) else int(round(v)) for i, v in enumerate(values)]
    return {
        "raw_repr": vals,
        "missing_indices": sorted(missing),
        "unit": "ADC",
        "coordinate_basis": "adc_code_8bit",
        "adc_full_scale": ADC_FS,
        "adc_baseline": ADC_BASE,
        "saturation": {"rail_high": ADC_FS, "rail_low": 0},
        "note": note,
    }


def segment(no, label, channel, grid, values, trigger_us, gain_raw="10",
            probe_raw="10x", deskew_raw="0", deskew_unit="ns",
            window=("w1", 8.0, 20.0), missing=(), note="", gain_unit="mV/ADC",
            deskew_basis="channel_deskew_vs_CH1"):
    wv, ws, we = window
    return {
        "segment_no": no,
        "label": label,
        "channel": channel,
        "note": note,
        "samples": samples_payload(grid, values, missing, note),
        "timebase": (
            q("1.0", "ns", "uniform_sample_interval")
            if grid == regular_grid(len(grid))
            else {
                "raw_repr": grid,
                "unit": "ns",
                "coordinate_basis": "sample_times_relative_to_segment_origin",
                "kind": "irregular",
            }
        ),
        "gain": q(gain_raw, gain_unit, "vertical_gain_per_adc_code"),
        "probe": {"raw_repr": probe_raw, "unit": "ratio",
                  "coordinate_basis": "scope_probe_attenuation"},
        "trigger_ts": q(str(trigger_us), "us", "capture_run_clock"),
        "deskew": q(deskew_raw, deskew_unit, deskew_basis),
        "measurement_window": {
            "version": wv,
            "start": q(str(ws), "ns", "relative_to_segment_origin"),
            "end": q(str(we), "ns", "relative_to_segment_origin"),
        },
    }


def build_case() -> dict:
    # S1 参考段 CH1：单峰居中 12ns
    g1 = regular_grid(44)
    v1 = gaussian(g1, [12.0], 120)
    s1 = segment(1, "S1 参考重复脉冲 CH1", "CH1", g1, v1, 100.000,
                 note="参考段，规则脉冲")

    # S2 正常重复段，事件物理位置 14ns，触发时标偏移 50ns -> 已知抖动 48ns
    g2 = regular_grid(44)
    v2 = gaussian(g2, [14.0], 120)
    s2 = segment(2, "S2 抖动重复段 CH1", "CH1", g2, v2, 100.050,
                 note="触发抖动 48ns（物理时间口径）")

    # S3 不等间隔时基：峰值物理位置 13ns，索引口径会误算
    g3 = irregular_grid()
    v3 = gaussian(g3, [13.0], 120)
    s3 = segment(3, "S3 不等间隔时基 CH1", "CH1", g3, v3, 100.100,
                 window=("w1", 8.0, 26.0), note="20ns 后采样间隔变为 2ns")

    # S4 竖直饱和：CH2，峰值削顶到满量程，饱和区间须排除
    g4 = regular_grid(44)
    v4 = gaussian(g4, [12.0], 150)
    s4 = segment(4, "S4 竖直饱和 CH2", "CH2", g4, v4, 100.150,
                 window=("w1", 8.0, 20.0), note="峰值削顶至 ADC 满量程")

    # S5 多峰：两个同幅脉冲，互相关产生两个等权合法候选
    g5 = regular_grid(44)
    v5 = gaussian(g5, [12.0, 22.0], 120)
    s5 = segment(5, "S5 多峰对齐 CH1", "CH1", g5, v5, 100.200,
                 window=("w1", 8.0, 26.0), note="两个等幅峰，对齐不唯一")

    # S6 CH2 deskew：格内峰值 12ns，通道 deskew +5ns
    g6 = regular_grid(44)
    v6 = gaussian(g6, [12.0], 120)
    s6 = segment(6, "S6 通道 deskew CH2", "CH2", g6, v6, 100.250,
                 deskew_raw="5", note="CH2 相对 CH1 deskew +5ns")

    # S7 缺样：峰值尾部 14,15 两采样缺失，缺样区间不进入插值
    g7 = regular_grid(44)
    v7 = gaussian(g7, [12.0], 120)
    s7 = segment(7, "S7 缺样 CH1", "CH1", g7, v7, 100.300,
                 missing=(14, 15), note="索引 14、15 采样缺失")

    # S8 触发在边界 + 无法转换的增益（保留原位，ADC 叠影仍可画）
    g8 = regular_grid(44)
    v8 = gaussian(g8, [3.0], 120)
    s8 = segment(8, "S8 触发在边界 CH1", "CH1", g8, v8, 100.350,
                 gain_raw="待标定", window=("w1", 0.0, 12.0),
                 note="峰贴近采集起点；增益未标定，电压不可用但 ADC 仍显示")

    # 测量区间版本修订（重放时由 w1 修订为 w2）
    window_revision = {
        "version": "w2",
        "start": q("6", "ns", "relative_to_segment_origin"),
        "end": q("18", "ns", "relative_to_segment_origin"),
        "reason": "测量区间收紧到脉冲主体",
    }

    rule_versions = {
        "derive": "derive-1.0.0",
        "measurement_window": "w1",
        "alignment": {"peak_prominence_ratio": 0.90, "min_overlap": 0.6},
        "note": "初始规则与参数版本",
    }

    narrative = [
        {"step": 1, "kind": "import",
         "text": "信号完整性团队导入 8 段分段捕获、触发标记与通道标定。"},
        {"step": 2, "kind": "verify_sources",
         "text": "核对每段原采样、时基、竖直增益、探头衰减、触发时标、deskew、测量区间版本。"},
        {"step": 3, "kind": "boundary_check",
         "text": "触发不等间隔时基/饱和/多峰/deskew/缺样/边界/区间修订边界组合。"},
        {"step": 4, "kind": "branch",
         "text": "建立第二分支，检验物理时间口径与多候选保留。"},
        {"step": 5, "kind": "parameter_revision",
         "text": "对齐参数修订（peak_prominence 0.90 -> 0.80），按依赖范围失效。"},
        {"step": 6, "kind": "local_source_change",
         "text": "只修订 S6 deskew 5ns -> 3ns，失效范围仅限该段链路。"},
        {"step": 7, "kind": "conflict",
         "text": "两个窗口同时裁定，后提交者看到 409 版本冲突与双方差异。"},
        {"step": 8, "kind": "review",
         "text": "页面查看叠影/抖动分布/触发候选/测量窗/排除原因，完成裁定。"},
        {"step": 9, "kind": "history",
         "text": "导出、删库、重导、重开，身份、顺序、来源关系保持。"},
    ]

    return {
        "case_id": CASE_ID,
        "title": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "domain": "signal_integrity_oscilloscope_segmented_capture",
        "channels": {
            "CH1": {"label": "通道1", "probe_default": "10x"},
            "CH2": {"label": "通道2", "probe_default": "10x",
                    "deskew_initial_ns": 5.0},
        },
        "rule_versions": rule_versions,
        "measurement_window_revision": window_revision,
        "segments": [s1, s2, s3, s4, s5, s6, s7, s8],
        "narrative": narrative,
        "expected": {
            "jitter_ns_by_segment": {"2": 46.0},
            "multi_peak_segment": 5,
            "saturated_segment": 4,
            "missing_segment": 7,
            "irregular_segment": 3,
            "deskew_segment": 6,
            "boundary_segment": 8,
            "unconvertible_segment": 8,
            "window_versions": ["w1", "w2"],
        },
    }


def write_case_pack(target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, "case.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_case(), fh, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "cases", "oscilloscope_lineage")
    print(write_case_pack(out))
