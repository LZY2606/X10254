#!/usr/bin/env python3
"""Build the deterministic signal-integrity offline case package."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "offline_case" / "waveform-overlay-lineage-v1"
FILE_DIR = CASE_DIR / "files"

SEGMENTS = {
    "seg_1": {"channel": "chA", "trigger_ns": 100.0, "amplitude": 0.8, "shift": 0.2, "style": "irregular"},
    "seg_2": {"channel": "chA", "trigger_ns": 200.0, "amplitude": 1.25, "shift": 0.4, "style": "uniform"},
    "seg_3": {"channel": "chC", "trigger_ns": 300.0, "amplitude": 0.8, "shift": None, "style": "uniform", "double": True},
    "seg_4": {"channel": "chD", "trigger_ns": 400.0, "amplitude": 0.8, "shift": -0.35, "style": "uniform"},
    "seg_5": {"channel": "chE", "trigger_ns": 500.0, "amplitude": 0.8, "shift": 0.7, "style": "missing"},
    "seg_6": {"channel": "chB", "trigger_ns": 600.0, "amplitude": 0.8, "shift": 0.3, "style": "uniform"},
    "seg_7": {"channel": "chA", "trigger_ns": 700.0, "amplitude": 0.8, "shift": 0.0, "style": "boundary"},
}

CHANNELS = {
    "chA": {"gain": "1.0 mV/count", "atten": "1x", "deskew": "0 ps"},
    "chB": {"gain": "800 µV/count", "atten": "10x", "deskew": "0 ps"},
    "chC": {"gain": "1.0 mV/count", "atten": "1x", "deskew": "0 ps"},
    "chD": {"gain": "1.0 mV/count", "atten": "1x", "deskew": "350 ps"},
    "chE": {"gain": "1.0 mV/count", "atten": "1x", "deskew": "0 ps"},
}


def sample_times(style: str) -> np.ndarray:
    if style == "irregular":
        return np.array([-3.2, -3.0, -2.7, -2.5, -2.2, -2.0, -1.7, -1.5,
                         -1.2, -1.0, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5,
                         0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5, 2.7, 3.0])
    if style == "boundary":
        return np.arange(0.0, 6.01, 0.3)
    return np.arange(-3.0, 3.01, 0.25)


def waveform(seg_id: str, times: np.ndarray) -> list[str]:
    spec = SEGMENTS[seg_id]
    rows: list[str] = []
    for idx, rel in enumerate(times):
        if spec["style"] == "missing" and abs(rel - 0.7) < 1e-12:
            rows.append("")
            continue
        if spec.get("double"):
            voltage = 0.78 * np.exp(-((rel + 0.8) / 0.42) ** 2) + 0.72 * np.exp(-((rel - 0.8) / 0.42) ** 2)
        else:
            voltage = spec["amplitude"] * np.exp(-((rel - spec["shift"]) / 0.42) ** 2)
        gain_v = 0.0008 if spec["channel"] == "chB" else 0.001
        if spec["channel"] == "chC":
            gain_v = 0.001
        code = int(round(voltage / gain_v))
        if seg_id == "seg_2" and abs(rel - 0.4) <= 0.5:
            code = 1200
        rows.append(str(code))
    return rows


def write_files() -> dict[str, dict]:
    FILE_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for seg_id, spec in SEGMENTS.items():
        times = sample_times(spec["style"])
        codes = waveform(seg_id, times)
        lines = ["segment_id,time_ns,voltage_code"]
        for time_value, code in zip(times, codes):
            lines.append(f"{seg_id},{format(time_value, '.12g')},{code}")
        path = f"files/{seg_id}_samples.csv"
        (FILE_DIR / f"{seg_id}_samples.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        files[path] = {"media_type": "text/csv", "role": "segment_samples", "segment_id": seg_id}

    triggers = ["segment_id,trigger_marker_ns,coordinate_basis"]
    for seg_id, spec in SEGMENTS.items():
        triggers.append(f"{seg_id},{format(spec['trigger_ns'], '.1f')},acquisition_clock/absolute_ns")
    path = "files/trigger_markers.csv"
    (FILE_DIR / "trigger_markers.csv").write_text("\n".join(triggers) + "\n", encoding="utf-8")
    files[path] = {"media_type": "text/csv", "role": "trigger_markers"}

    timebases = ["segment_id,index,time_ns,interval_kind"]
    for seg_id, spec in SEGMENTS.items():
        times = sample_times(spec["style"])
        for idx, time_value in enumerate(times):
            kind = "uniform" if spec["style"] in {"uniform", "boundary"} else "nonuniform"
            if spec["style"] == "missing" and idx:
                kind = "nonuniform"
            timebases.append(f"{seg_id},{idx},{format(time_value, '.12g')},{kind}")
    path = "files/timebases.csv"
    (FILE_DIR / "timebases.csv").write_text("\n".join(timebases) + "\n", encoding="utf-8")
    files[path] = {"media_type": "text/csv", "role": "timebases"}

    for channel, cal in CHANNELS.items():
        text = json.dumps({
            "channel_id": channel,
            "vertical_gain": {"raw_representation": cal["gain"], "unit": "V/count"},
            "probe_attenuation": {"raw_representation": cal["atten"], "unit": "ratio"},
            "deskew": {"raw_representation": cal["deskew"], "coordinate_basis": "channel/reference_trigger"},
        }, ensure_ascii=False, indent=2)
        path = f"files/{channel}_calibration.json"
        (FILE_DIR / f"{channel}_calibration.json").write_text(text + "\n", encoding="utf-8")
        files[path] = {"media_type": "application/json", "role": "channel_calibration", "channel_id": channel}

    window = {
        "window_id": "meas_window",
        "version": "v1",
        "start_ns": -0.4,
        "end_ns": 2.0,
        "coordinate_basis": "trigger_relative_ns",
        "quantity": "pulse_area",
    }
    path = "files/measurement_window_v1.json"
    (FILE_DIR / "measurement_window_v1.json").write_text(json.dumps(window, indent=2) + "\n", encoding="utf-8")
    files[path] = {"media_type": "application/json", "role": "measurement_window"}
    return files


def material_records() -> list[dict]:
    records = []
    for seg_id, spec in SEGMENTS.items():
        records.append({
            "material_id": f"{seg_id}_samples",
            "material_type": "segment_samples",
            "path": f"files/{seg_id}_samples.csv",
            "segment_id": seg_id,
            "channel_id": spec["channel"],
            "raw_representation": {"columns": ["time_ns", "voltage_code"], "time_unit": "ns", "code_unit": "ADC count"},
            "coordinate_basis": "trigger_relative_ns",
        })
        records.append({
            "material_id": f"{seg_id}_timebase",
            "material_type": "timebase",
            "path": "files/timebases.csv",
            "segment_id": seg_id,
            "raw_representation": {"column": "time_ns", "unit": "ns"},
            "coordinate_basis": "trigger_relative_ns",
        })
        records.append({
            "material_id": f"{seg_id}_trigger",
            "material_type": "trigger_timestamp",
            "path": "files/trigger_markers.csv",
            "segment_id": seg_id,
            "raw_representation": {"column": "trigger_marker_ns", "unit": "ns"},
            "coordinate_basis": "acquisition_clock/absolute_ns",
        })
        records.append({
            "material_id": f"{seg_id}_calibration",
            "material_type": "channel_calibration",
            "path": f"files/{spec['channel']}_calibration.json",
            "segment_id": seg_id,
            "channel_id": spec["channel"],
            "raw_representation": {"vertical_gain": CHANNELS[spec["channel"]]["gain"],
                                   "probe_attenuation": CHANNELS[spec["channel"]]["atten"],
                                   "deskew": CHANNELS[spec["channel"]]["deskew"]},
        })
    records.append({
        "material_id": "meas_window_v1",
        "material_type": "measurement_window",
        "path": "files/measurement_window_v1.json",
        "window_id": "meas_window",
        "version": "v1",
        "raw_representation": {"start": "-0.4 ns", "end": "2.0 ns", "basis": "trigger_relative_ns"},
    })
    return records


def build() -> None:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    files = write_files()
    embedded = {}
    for path, meta in sorted(files.items()):
        data = (CASE_DIR / path).read_bytes()
        embedded[path] = dict(meta, bytes_base64=base64.b64encode(data).decode("ascii"),
                              sha256=hashlib.sha256(data).hexdigest())
    bundle = {
        "schema_version": "1",
        "case_id": "waveform-overlay-lineage",
        "title": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "team": "信号完整性团队",
        "narrative": [
            "导入分段捕获、触发标记与通道标定。",
            "核对每段原采样、时基、竖直增益、探头衰减、触发时标、deskew 和测量窗口版本。",
            "在第二分支保留多峰候选并审查不等间隔、饱和、deskew、缺样和触发边界诊断。",
        ],
        "materials": material_records(),
        "files": embedded,
        "fixed_situations": [
            "nonuniform_timebase", "vertical_saturation", "multipeak_alignment",
            "channel_deskew", "missing_samples", "trigger_at_boundary", "measurement_window_revision",
        ],
        "revisions": {
            "window_v2": {
                "material": {
                    "material_id": "meas_window_v2",
                    "material_type": "measurement_window",
                    "window_id": "meas_window",
                    "version": "v2",
                    "raw_representation": {"start": "0.0 ns", "end": "2.2 ns", "basis": "trigger_relative_ns"},
                    "coordinate_basis": "trigger_relative_ns",
                },
                "expected_invalidated_families": ["measurement"],
            },
            "chC_gain_fix": {
                "material": {
                    "material_id": "cal_C_v2",
                    "material_type": "channel_calibration",
                    "channel_id": "chC",
                    "raw_representation": {"vertical_gain": "1.2 mV/count", "probe_attenuation": "1x", "deskew": "0 ps"},
                },
                "expected_invalidated_segments": ["seg_3"],
            },
        },
    }
    bundle_bytes = json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    (CASE_DIR / "case_bundle.json").write_bytes(bundle_bytes + b"\n")
    readme = """# 波形叠影台离线案例\n\n固定案例包含七段分段捕获，覆盖不等间隔时基、竖直饱和、多峰对齐、通道 deskew、缺样、触发边界与测量窗口修订。\n`case_bundle.json` 内嵌全部物理 CSV/JSON 字节；导入只把原始表示放入来源层，规范化结果进入派生层。\n"""
    (CASE_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    build()
