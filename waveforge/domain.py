from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np

TIME_UNITS_TO_S = {
    "ps": 1e-12,
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
}
GAIN_UNITS_TO_V_PER_COUNT = {
    "v/count": 1.0,
    "mv/count": 1e-3,
    "uv/count": 1e-6,
}


def stable_id(*parts: str) -> str:
    return ":".join(str(part) for part in parts)


def parse_seconds(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, dict):
        raw_value = value.get("value")
        unit = str(value.get("unit", "")).strip()
        display = value.get("display")
    else:
        raw_value = value
        unit = "s"
        display = None
    if raw_value is None:
        return None, "MISSING_NUMERIC_VALUE"
    try:
        number = float(raw_value)
    except (TypeError, ValueError):
        return None, "NON_NUMERIC_TIME"
    factor = TIME_UNITS_TO_S.get(unit.lower())
    if factor is None:
        return None, "UNKNOWN_TIME_UNIT"
    if not np.isfinite(number):
        return None, "NON_FINITE_TIME"
    return number * factor, None


def parse_volts_per_count(value: dict[str, Any]) -> tuple[float | None, str | None]:
    raw_value = value.get("value") if isinstance(value, dict) else None
    unit = str(value.get("unit", "")).strip().lower() if isinstance(value, dict) else ""
    try:
        number = float(raw_value)
    except (TypeError, ValueError):
        return None, "NON_NUMERIC_VERTICAL_GAIN"
    factor = GAIN_UNITS_TO_V_PER_COUNT.get(unit)
    if factor is None:
        return None, "UNKNOWN_GAIN_UNIT"
    if number <= 0 or not np.isfinite(number):
        return None, "INVALID_VERTICAL_GAIN"
    return number * factor, None


def parse_probe_ratio(value: dict[str, Any] | str | None) -> tuple[float | None, str | None]:
    if isinstance(value, dict):
        display = str(value.get("display", value.get("value", "")))
    else:
        display = str(value)
    text = display.strip().lower().replace("×", "x").replace(" ", "")
    if text.startswith("x"):
        text = text[1:]
    if text.endswith(":1"):
        text = text[:-2]
    try:
        ratio = float(Decimal(text))
    except (InvalidOperation, ValueError):
        return None, "UNPARSEABLE_PROBE_ATTENUATION"
    if ratio <= 0 or not np.isfinite(ratio):
        return None, "INVALID_PROBE_ATTENUATION"
    return ratio, None


def parse_offset_volts(value: dict[str, Any]) -> tuple[float | None, str | None]:
    if not isinstance(value, dict) or "value" not in value:
        return None, "OFFSET_WITHOUT_NUMERIC_VALUE"
    raw_value = value.get("value")
    unit = str(value.get("unit", "")).lower()
    try:
        number = float(raw_value)
    except (TypeError, ValueError):
        return None, "NON_NUMERIC_OFFSET"
    if unit == "mv":
        return number * 1e-3, None
    if unit == "v":
        return number, None
    return None, "UNKNOWN_OFFSET_UNIT"


def material_timebase_seconds(timebase: dict[str, Any], count: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    if timebase.get("type") == "uniform":
        origin, code = parse_seconds(timebase.get("origin", {"value": 0, "unit": "s"}))
        increment, inc_code = parse_seconds(timebase.get("increment"))
        if origin is None or increment is None:
            raise ValueError(code or inc_code or "INVALID_UNIFORM_TIMEBASE")
        if increment <= 0:
            raise ValueError("NON_POSITIVE_TIMEBASE_INCREMENT")
        return np.arange(count, dtype=float) * increment + origin, diagnostics
    raw_points = timebase.get("points_seconds") or []
    times: list[float] = []
    for index, point in enumerate(raw_points):
        seconds, code = parse_seconds(point)
        if seconds is None:
            diagnostics.append({
                "code": "UNCONVERTIBLE_TIME_SAMPLE",
                "severity": "error",
                "index": index,
                "raw": point,
                "message": f"time sample remains at index {index} with diagnostic {code}",
            })
            continue
        times.append(seconds)
    if not times:
        raise ValueError("NO_CONVERTIBLE_TIME_SAMPLES")
    if np.any(np.diff(np.asarray(times)) <= 0):
        diagnostics.append({
            "code": "NON_MONOTONIC_TIMEBASE",
            "severity": "error",
            "message": "time points must be strictly increasing; no index subtraction was used",
        })
    return np.asarray(times, dtype=float), diagnostics


def interval_contains(interval_seconds: dict[str, Any], point_seconds: float) -> bool:
    start, _ = parse_seconds(interval_seconds["start"])
    end, _ = parse_seconds(interval_seconds["end"])
    if start is None or end is None or end < start:
        return False
    return start - 1e-12 <= point_seconds <= end + 1e-12


def build_channel_series(
    segment: dict[str, Any],
    channel_id: str,
    channel: dict[str, Any],
) -> dict[str, Any]:
    raw = segment["channels"][channel_id].get("raw_samples", [])
    raw_times = segment.get("timebase", {})
    times, time_diag = material_timebase_seconds(raw_times, len(raw))
    # Raw samples align to convertible non-uniform timestamps. Unconvertible timestamps
    # are retained in the imported material and excluded from the derived array.
    if raw_times.get("type") != "uniform":
        convertible = [idx for idx, point in enumerate(raw_times.get("points_seconds", []))
                       if parse_seconds(point)[0] is not None]
        raw = [raw[i] for i in convertible if i < len(raw)]
    raw = list(raw[:len(times)])

    gain, gain_code = parse_volts_per_count(channel["vertical_gain"])
    probe, probe_code = parse_probe_ratio(channel.get("probe_attenuation"))
    offset, offset_code = parse_offset_volts(channel.get("vertical_offset", {"value": 0, "unit": "mV"}))
    deskew, deskew_code = parse_seconds(channel.get("deskew", {"value": 0, "unit": "ns"}))
    diagnostics = list(time_diag)
    if gain is None:
        diagnostics.append({"code": gain_code, "severity": "error", "where": "vertical_gain"})
    if probe is None:
        diagnostics.append({"code": probe_code, "severity": "error", "where": "probe_attenuation"})
    if offset is None:
        diagnostics.append({"code": offset_code, "severity": "warning", "where": "vertical_offset"})
        offset = 0.0
    if deskew is None:
        diagnostics.append({"code": deskew_code, "severity": "error", "where": "deskew"})
        deskew = 0.0

    missing = np.asarray([sample is None for sample in raw], dtype=bool)
    numeric = np.asarray([0.0 if sample is None else float(sample) for sample in raw], dtype=float)
    saturated = np.zeros(len(raw), dtype=bool)
    raw_saturated = segment["channels"][channel_id].get("saturated_intervals", [])
    # Saturation is evaluated on the actual unequal time axis, never by index range.
    for interval in raw_saturated:
        for index, t in enumerate(times):
            if interval_contains(interval, t):
                saturated[index] = True
    valid = (~missing) & (~saturated)
    volts = np.full(len(raw), np.nan, dtype=float)
    if gain is not None and probe is not None:
        volts[valid] = numeric[valid] * gain * probe - offset
    corrected_time = times - float(deskew)
    return {
        "raw_samples": raw,
        "time_seconds": corrected_time,
        "raw_time_seconds": times,
        "volts": volts,
        "missing": missing,
        "saturated": saturated,
        "valid": valid,
        "diagnostics": diagnostics,
        "gain_v_per_count": gain,
        "probe_ratio": probe,
        "offset_v": offset,
        "deskew_s": deskew,
    }


def _score_pair(ref_t: np.ndarray, ref_y: np.ndarray, seg_t: np.ndarray,
                seg_y: np.ndarray, shift: float, tolerance: float = 1e-12) -> dict[str, Any] | None:
    # Exact time-coordinate overlap only. Missing and saturated values are masked and
    # no interpolation is performed to manufacture overlap.
    active = np.isfinite(seg_y) & (np.abs(seg_y) > 0.025)
    if not np.any(active):
        return None
    support_min = float(np.min(seg_t[active]) - shift)
    support_max = float(np.max(seg_t[active]) - shift)
    ref_map = {round(float(t) / tolerance): float(y)
               for t, y in zip(ref_t, ref_y)
               if np.isfinite(y) and support_min <= t <= support_max}
    pairs_a: list[float] = []
    pairs_b: list[float] = []
    for t, y in zip(seg_t, seg_y):
        if not np.isfinite(y) or abs(y) <= 0.025:
            continue
        key = round(float(t - shift) / tolerance)
        if key in ref_map:
            pairs_a.append(ref_map[key])
            pairs_b.append(float(y))
    active_pair_candidates = int(np.count_nonzero(active))
    if len(pairs_a) < 3 or len(pairs_a) / max(active_pair_candidates, 1) < 0.9:
        return None
    a = np.asarray(pairs_a)
    b = np.asarray(pairs_b)
    if np.std(a) <= 1e-9 or np.std(b) <= 1e-9:
        return None
    shape = float(np.corrcoef(a, b)[0, 1])
    normalizer = 4.8
    rmse = float(np.sqrt(np.mean((a - b) ** 2))) / normalizer
    score = max(0.0, 0.70 * float(np.exp(-rmse / 0.16)) + 0.30 * shape)
    if np.allclose(a, b, atol=1e-7, rtol=1e-7):
        score = 1.0
    if not np.isfinite(score):
        score = 0.0
    return {"shift_seconds": float(shift), "score": float(max(0.0, min(1.0, score))), "pair_count": len(a)}


def alignment_candidates(reference: dict[str, Any], target: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    ref_t = reference["time_seconds"]
    seg_t = target["time_seconds"]
    shifts = np.unique(np.round((seg_t[:, None] - ref_t[None, :]).ravel(), 12))
    candidates = []
    for shift in shifts:
        scored = _score_pair(ref_t, reference["volts"], seg_t, target["volts"], float(shift))
        if scored and scored["score"] + 1e-12 >= threshold:
            candidates.append(scored)
    candidates.sort(key=lambda item: (-item["score"], item["shift_seconds"]))
    rank = 0
    for item in candidates:
        rank += 1
        item["candidate_id"] = stable_id("cand", "lag", f"{item['shift_seconds']:.12g}s")
        item["rank"] = rank
        item["status"] = "legal_ambiguous" if len(candidates) > 1 else "legal_unique"
    return candidates


def measurement_in_window(series: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    start, start_code = parse_seconds(window["start"])
    end, end_code = parse_seconds(window["end"])
    diagnostics: list[dict[str, Any]] = []
    if start is None or end is None or end < start:
        return {"status": "INVALID_WINDOW", "value_v": None, "diagnostics": [
            {"code": start_code or end_code or "WINDOW_START_AFTER_END", "severity": "error"}]}
    times = series["time_seconds"]
    volts = series["volts"]
    valid = series["valid"]
    inside = (times >= start - 1e-12) & (times <= end + 1e-12)
    selected = inside & valid
    missing_count = int(np.count_nonzero(inside & series["missing"]))
    saturated_count = int(np.count_nonzero(inside & series["saturated"]))
    if missing_count:
        diagnostics.append({"code": "MISSING_SAMPLES_EXCLUDED", "severity": "warning", "count": missing_count})
    if saturated_count:
        diagnostics.append({"code": "SATURATED_SAMPLES_EXCLUDED", "severity": "warning", "count": saturated_count})
    if np.count_nonzero(selected) < 2:
        return {"status": "INSUFFICIENT_VALID_SAMPLES", "value_v": None, "diagnostics": diagnostics}
    values = volts[selected]
    peak_index = int(np.argmax(values))
    peak_time = float(times[selected][peak_index])
    # A same-height second maximum is deliberately retained as an alignment/measurement ambiguity.
    maximum = float(values[peak_index])
    near_max = np.abs(values - maximum) < 1e-7
    if int(np.count_nonzero(near_max)) > 1:
        diagnostics.append({"code": "MULTIPLE_PEAKS_IN_WINDOW", "severity": "warning",
                            "count": int(np.count_nonzero(near_max))})
    return {
        "status": "MEASURED",
        "value_v": maximum,
        "peak_time_seconds": peak_time,
        "sample_count": int(np.count_nonzero(selected)),
        "diagnostics": diagnostics,
    }


@dataclass(frozen=True)
class DependencyNode:
    kind: str
    identity: str

    def key(self) -> str:
        return stable_id("dep", self.kind, self.identity)


def segment_nodes(segment_id: str) -> set[DependencyNode]:
    return {
        DependencyNode("segment_samples", segment_id),
        DependencyNode("trigger", segment_id),
        DependencyNode("timebase", segment_id),
    }


def channel_nodes(channel_id: str, segment_id: str | None = None) -> set[DependencyNode]:
    suffix = channel_id if segment_id is None else stable_id(segment_id, channel_id)
    return {
        DependencyNode("vertical_gain", suffix),
        DependencyNode("probe_attenuation", suffix),
        DependencyNode("deskew", suffix),
    }


def rule_nodes(rule_id: str) -> set[DependencyNode]:
    return {DependencyNode("rule", rule_id)}


def window_nodes(window_id: str) -> set[DependencyNode]:
    return {DependencyNode("measurement_window", window_id)}


def affected_scope(dependencies: list[str], changed_nodes: set[str]) -> list[str]:
    changed = {node if isinstance(node, str) else node.key() for node in changed_nodes}
    return sorted({dep for dep in dependencies if dep in changed})
