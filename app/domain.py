from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any

import numpy as np


UNIT_TO_NS = {"ns": 1.0, "us": 1000.0, "µs": 1000.0, "s": 1_000_000_000.0}


def stable_id(prefix: str, *parts: Any) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def measured(value: dict[str, Any], unit: str, coordinate_basis: str, display_id: str,
             source_refs: list[str], dependencies: list[str], uncertainty: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "display_id": display_id,
        "value": value,
        "unit": unit,
        "coordinate_basis": coordinate_basis,
        "source_refs": list(source_refs),
        "dependencies": list(dependencies),
        "uncertainty": uncertainty or {},
    }


def to_nanoseconds(value: Any, unit: str) -> float | None:
    try:
        return float(value) * UNIT_TO_NS[unit]
    except (TypeError, ValueError, KeyError):
        return None


def number_from(value: dict[str, Any]) -> float | None:
    if not isinstance(value, dict):
        return None
    try:
        return float(value.get("value"))
    except (TypeError, ValueError):
        return None


def raw_material(raw: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(raw)


def _material_refs(materials: dict[str, dict[str, Any]]) -> list[str]:
    return [f"material:{key}:{item['id']}" for key, item in sorted(materials.items())]


def _all_segments(state: dict[str, Any]) -> list[str]:
    return sorted({
        key.split(":", 1)[1]
        for key in state["materials"]
        if key.startswith("segment:")
    })


def _current(state: dict[str, Any], key: str) -> dict[str, Any]:
    return state["materials"][key]


def material_digest(material: dict[str, Any]) -> str:
    raw = material["raw"]
    return hashlib.sha256(
        __import__("json", fromlist=["dumps"]).dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def initial_state(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "materials": {},
        "material_history": [],
        "decisions": {},
        "decision_history": [],
        "derived": {},
        "invalidations": [],
        "judgments": [],
        "parameter_version": "alignment-parameters-v1",
        "window_version": "measurement-window-v1",
    }


def _upsert_material(state: dict[str, Any], material: dict[str, Any]) -> str | None:
    key = material["stable_key"]
    old = state["materials"].get(key)
    if old and material_digest(old) == material_digest(material):
        return None
    state["materials"][key] = material
    state["material_history"].append({
        "stable_key": key,
        "material_id": material["id"],
        "supersedes": old["id"] if old else None,
        "digest": material_digest(material),
    })
    return old["id"] if old else None


def apply_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(state)
    state["judgments"] = []
    state["invalidations"] = []
    payload = event["payload"]
    event_type = event["type"]

    if event_type == "case_imported":
        state["case_id"] = payload["case_id"]
        for material in payload["materials"]:
            _upsert_material(state, material)
    elif event_type in {"parameter_revised", "source_amended", "measurement_window_revised"}:
        material = payload["material"]
        key = material["stable_key"]
        old_id = _upsert_material(state, material)
        if key == "parameter:alignment":
            state["parameter_version"] = material["raw"]["version"]
        if key.startswith("window:"):
            state["window_version"] = material["raw"]["version"]
        state["invalidations"] = _invalidation_range(state, key, material, old_id, event)
    elif event_type == "manual_decision":
        candidate_id = payload["candidate_id"]
        previous = state["decisions"].get(candidate_id)
        state["decisions"][candidate_id] = {
            "candidate_id": candidate_id,
            "status": payload["status"],
            "reason": payload.get("reason", ""),
            "event_id": event["id"],
            "branch_id": event["branch_id"],
            "supersedes_decision_event": previous["event_id"] if previous else None,
        }
        state["decision_history"].append(state["decisions"][candidate_id])
    else:
        raise ValueError(f"unknown event type {event_type}")

    state["derived"] = recompute_derived(state, event)
    return state


def _invalidation_range(state: dict[str, Any], key: str, material: dict[str, Any],
                        old_id: str | None, event: dict[str, Any]) -> list[dict[str, Any]]:
    if key == "parameter:alignment":
        targets = [f"alignment:{segment}" for segment in _all_segments(state)]
        scope = "alignment_parameter"
    elif key.startswith("window:"):
        targets = [f"measurement:{segment}" for segment in _all_segments(state)] + ["jitter:distribution"]
        scope = "measurement_window_version"
    elif key.startswith("trigger:"):
        segment_id = key.split(":", 1)[1]
        targets = [f"jitter:{segment_id}", "jitter:distribution"]
        scope = "local_trigger_source"
    elif key.startswith("segment:"):
        segment_id = key.split(":", 1)[1]
        targets = [f"alignment:{segment_id}", f"measurement:{segment_id}", f"jitter:{segment_id}", "jitter:distribution"]
        scope = "local_segment_source"
    elif key.startswith("calibration:"):
        targets = []
        for segment in _all_segments(state):
            targets.extend([f"alignment:{segment}", f"measurement:{segment}"])
        targets.append("jitter:distribution")
        scope = "channel_calibration"
    else:
        targets = []
        scope = "unknown"
    return [{
        "display_id": stable_id("invalidation", event["id"], target),
        "target_id": target,
        "scope": scope,
        "changed_material_id": material["id"],
        "previous_material_id": old_id,
        "event_id": event["id"],
        "replaced_globally": False,
        "reason": "dependency-graph recomputation; historical snapshots retain prior conclusions",
    } for target in targets]


def _calibration(state: dict[str, Any], channel_id: str) -> dict[str, Any]:
    return state["materials"][f"calibration:{channel_id}"]["raw"]


def _convert_samples(material: dict[str, Any], calibration: dict[str, Any]) -> list[dict[str, Any]]:
    gain = float(calibration["vertical_scale"]["value"])
        attenuation = float(calibration["probe_attenuation"]["ratio"])
    offset = float(calibration.get("vertical_offset", {"value": 0})["value"])
    converted = []
    for point in material["raw"]["samples"]:
        adc = point.get("adc_count")
        quality = list(point.get("quality", []))
        volts = None
        if adc is not None and "missing" not in quality:
            volts = adc * gain * attenuation + offset
        converted.append({
            "time": point["time"],
            "time_ns": to_nanoseconds(point["time"]["value"], point["time"]["unit"]),
            "adc_count": adc,
            "volts": volts,
            "quality": quality,
            "source_sample_index": point["index"],
        })
    return converted


def _valid_grid(points: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    usable = [p for p in points if p["volts"] is not None and p["time_ns"] is not None
              and "missing" not in p["quality"] and "saturated" not in p["quality"]]
    return (np.array([p["time_ns"] for p in usable], dtype=float),
            np.array([p["volts"] for p in usable], dtype=float))


def _safe_interp(query_times: np.ndarray, valid_times: np.ndarray, valid_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(valid_times) < 2:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(valid_times)
    valid_times = valid_times[order]
    valid_values = valid_values[order]
    # np.interp is used only inside the convex hull of adjacent unsaturated, non-missing samples.
    # Exact endpoint membership is deliberately excluded so a missing/saturated boundary cannot enter.
    in_safe_span = (query_times > valid_times[0]) & (query_times < valid_times[-1])
    values = np.interp(query_times[in_safe_span], valid_times, valid_values)
    return values, in_safe_span


def _correlation_scores(ch1: list[dict[str, Any]], ch2: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    ch1_times, ch1_values = _valid_grid(ch1)
    ch2_times, ch2_values = _valid_grid(ch2)
    candidate_times = ch1_times
    shifted, safe = _safe_interp(candidate_times - _deskew_grid(ch2), ch2_times, ch2_values)
    base = candidate_times[safe]
    left = ch1_values[safe]
    if len(base) < 3:
        return np.array([]), np.array([])
    scores = []
    lags = []
    center = 5.0
    for lag_ns in np.arange(-3.0, 3.0001, 0.25):
        right, right_safe = _safe_interp(base - lag_ns, ch2_times, ch2_values)
        if len(right) < 3:
            continue
        x = left[: len(right)]
        if np.std(x) == 0 or np.std(right) == 0:
            score = 0.0
        else:
            score = float(np.corrcoef(x - x.mean(), right - right.mean())[0, 1])
        scores.append(score)
        lags.append(lag_ns)
    _ = center
    return np.array(lags), np.array(scores)


def _deskew_grid(points: list[dict[str, Any]]) -> float:
    return 0.0


def _peak_candidates(lags: np.ndarray, scores: np.ndarray, threshold: float,
                     segment_id: str) -> list[dict[str, float | str | bool]]:
    candidates: list[dict[str, Any]] = []
    if len(scores) == 0:
        return candidates
    peaks = []
    for i, score in enumerate(scores):
        if i == 0 or i == len(scores) - 1:
            is_peak = False
        else:
            is_peak = score > scores[i - 1] and score > scores[i + 1] and score >= threshold
        if is_peak:
            peaks.append((i, float(lags[i]), float(score)))
    if not peaks:
        best = int(np.argmax(scores))
        peaks = [(best, float(lags[best]), float(scores[best]))]
    for rank, (_, lag, score) in enumerate(sorted(peaks, key=lambda x: (-x[2], x[1]))[:3], start=1):
        candidates.append({
            "display_id": f"cand:{segment_id}:peak-{rank}",
            "lag_ns": round(lag, 3),
            "score": round(score, 6),
            "rank": rank,
            "legal": score >= threshold,
            "status": "proposed",
        })
    return candidates
