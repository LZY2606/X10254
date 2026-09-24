"""Import segmentation captures, trigger markers and channel calibration.

Source vs interpretation is split here: the importer stores the verbatim
source bytes (a content-addressed blob) plus a parsed view. No irreversible
cleaning happens at import; unconvertible values stay in place with a
diagnostic. A batch is validated before any row is written so that failure
rolls back the whole change (see :mod:`wavebench.store`).
"""
from __future__ import annotations

from typing import Any

from .canonical import digest, dumps
from .units import as_value, quantity, raw_repr

SCHEMA_KIND = "waveform-overlay-bench.package/v1"


class IngestError(ValueError):
    """Raised for a whole-batch validation failure."""


def _require(obj: dict[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise IngestError(f"{where}: missing required field {key!r}")
    return obj[key]


def _parse_segment(raw: dict[str, Any]) -> dict[str, Any]:
    seg_id = _require(raw, "segment_id", "segment")
    channel = _require(raw, "channel", f"segment {seg_id}")
    codes = _require(raw, "samples", f"segment {seg_id}")
    if not isinstance(codes, list) or not codes:
        raise IngestError(f"segment {seg_id}: samples must be a non-empty list")
    if not all(isinstance(c, (int, float)) or c is None for c in codes):
        raise IngestError(f"segment {seg_id}: samples must be numbers or null")

    tb_raw = _require(raw, "timebase", f"segment {seg_id}")
    tb_kind = _require(tb_raw, "type", f"segment {seg_id} timebase")
    parsed_tb: dict[str, Any] = {"raw": tb_raw}
    if tb_kind == "uniform":
        dt = quantity(_require(tb_raw, "dt", f"segment {seg_id} timebase"),
                      f"{seg_id}.timebase.dt", "time")
        parsed_tb["type"] = "uniform"
        parsed_tb["dt"] = dt
        if dt.get("value") is None:
            parsed_tb["diagnostic"] = dt.get("diagnostic")
    elif tb_kind == "nonuniform":
        times = _require(tb_raw, "times_ns", f"segment {seg_id} timebase")
        if not isinstance(times, list) or len(times) != len(codes):
            raise IngestError(
                f"segment {seg_id}: nonuniform times_ns length must match samples"
            )
        parsed = [quantity(t, f"{seg_id}.timebase.times_ns[{i}]", "time")
                  for i, t in enumerate(times)]
        parsed_tb["type"] = "nonuniform"
        parsed_tb["times"] = parsed
        deltas = [
            round(parsed[i + 1]["value"] - parsed[i]["value"], 18)
            for i in range(len(parsed) - 1)
            if parsed[i]["value"] is not None and parsed[i + 1]["value"] is not None
        ]
        if deltas:
            span = max(deltas) - min(deltas)
            if span <= 1e-15:
                parsed_tb["uniform_warning"] = (
                    "timebase declared nonuniform but spacings are equal"
                )
    else:
        raise IngestError(f"segment {seg_id}: unknown timebase type {tb_kind!r}")

    trig = quantity(_require(raw, "trigger_ts_ns", f"segment {seg_id}"),
                    f"{seg_id}.trigger_ts_ns", "time")
    missing = raw.get("missing_indexes", []) or []
    if not isinstance(missing, list):
        raise IngestError(f"segment {seg_id}: missing_indexes must be a list")

    parsed: dict[str, Any] = {
        "segment_id": seg_id,
        "channel": channel,
        "samples": list(codes),
        "timebase": parsed_tb,
        "trigger_ts_ns": trig,
        "missing_indexes": sorted(int(i) for i in missing),
        "notes": raw.get("notes"),
    }
    if "saturation_codes" in raw:
        parsed["saturation_codes"] = raw["saturation_codes"]
    if "t0_ns" in raw:
        parsed["t0_ns"] = quantity(raw["t0_ns"], f"{seg_id}.t0_ns", "time")
    return parsed


def _parse_trigger(raw: dict[str, Any]) -> dict[str, Any]:
    seg_id = _require(raw, "segment_id", "trigger_marker")
    ts = quantity(_require(raw, "ts_ns", f"trigger_marker {seg_id}"),
                  f"trigger.{seg_id}.ts_ns", "time")
    return {
        "segment_id": seg_id,
        "channel": _require(raw, "channel", f"trigger_marker {seg_id}"),
        "ts_ns": ts,
        "boundary_hint": raw.get("boundary_hint", False),
        "raw": raw,
    }


def _parse_calibration(raw: dict[str, Any]) -> dict[str, Any]:
    channel = _require(raw, "channel", "channel_calibration")
    gain_raw = _require(raw, "vertical_gain", f"calibration {channel}")
    gain = quantity(gain_raw, f"calibration.{channel}.vertical_gain")
    probe = quantity(_require(raw, "probe_attenuation",
                              f"calibration {channel}"),
                     f"calibration.{channel}.probe_attenuation", "ratio")
    deskew = quantity(_require(raw, "deskew", f"calibration {channel}"),
                      f"calibration.{channel}.deskew", "time")
    offset = quantity(raw.get("vertical_offset", 0),
                      f"calibration.{channel}.vertical_offset", "voltage")
    return {
        "channel": channel,
        "vertical_gain": gain,
        "probe_attenuation": probe,
        "deskew": deskew,
        "vertical_offset": offset,
        "raw": raw,
    }


def _parse_window_set(raw: dict[str, Any]) -> dict[str, Any]:
    version = _require(raw, "version", "measurement_window_set")
    windows_raw = _require(raw, "windows", f"window set {version}")
    windows = []
    for index, win in enumerate(windows_raw):
        wid = win.get("window_id", f"w{index}")
        start = quantity(_require(win, "start_ns", f"window {wid}"),
                         f"window.{version}.{wid}.start_ns", "time")
        end = quantity(_require(win, "end_ns", f"window {wid}"),
                       f"window.{version}.{wid}.end_ns", "time")
        if start["value"] is not None and end["value"] is not None \
                and end["value"] < start["value"]:
            raise IngestError(f"window {wid}: end_ns precedes start_ns")
        windows.append({
            "window_id": wid,
            "start_ns": start,
            "end_ns": end,
            "metric": win.get("metric", "amplitude"),
            "raw": win,
        })
    return {"version": version, "windows": windows, "raw": raw}


def parse_package(package: dict[str, Any]) -> dict[str, Any]:
    """Validate + parse one import batch. Raises IngestError on any problem."""
    kind = package.get("schema")
    if kind != SCHEMA_KIND:
        raise IngestError(
            f"unsupported schema {kind!r}; expected {SCHEMA_KIND!r}"
        )
    segments = _require(package, "segments", "package")
    if not isinstance(segments, list) or not segments:
        raise IngestError("package.segments must be a non-empty list")
    parsed_segments = [_parse_segment(s) for s in segments]
    seg_ids = [s["segment_id"] for s in parsed_segments]
    if len(set(seg_ids)) != len(seg_ids):
        raise IngestError("duplicate segment_id within batch")

    parsed_triggers = [_parse_trigger(t)
                       for t in package.get("trigger_markers", [])]
    parsed_calibrations = [_parse_calibration(c)
                           for c in package.get("channel_calibrations", [])]
    channels = {c["channel"] for c in parsed_calibrations}
    seg_channels = {s["channel"] for s in parsed_segments}
    missing_cal = sorted(seg_channels - channels)
    if missing_cal:
        raise IngestError(
            f"channel calibration missing for channels {missing_cal}"
        )
    window_sets = [_parse_window_set(w)
                   for w in package.get("measurement_window_sets", [])]
    if not window_sets:
        raise IngestError("package must version at least one measurement window set")

    return {
        "schema": kind,
        "segments": parsed_segments,
        "trigger_markers": parsed_triggers,
        "channel_calibrations": parsed_calibrations,
        "measurement_window_sets": window_sets,
        "raw": package,
    }


def build_materials(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a parsed package into material rows with stable identities."""
    materials: list[dict[str, Any]] = []

    def add(kind: str, key: str, parsed_obj: dict[str, Any], raw_obj: Any,
            extra: dict[str, Any] | None = None) -> None:
        identity = digest({"kind": kind, "key": key, "raw": raw_obj})
        row = {
            "id": identity,
            "kind": kind,
            "key": key,
            "parsed": parsed_obj,
            "raw": raw_obj,
            "hash": digest(raw_obj),
        }
        if extra:
            row.update(extra)
        materials.append(row)

    for seg in parsed["segments"]:
        add("segment", seg["segment_id"], seg, parsed["raw_refs"][seg["segment_id"]])
    for trig in parsed["trigger_markers"]:
        add("trigger_marker", trig["segment_id"], trig, trig["raw"])
    for cal in parsed["channel_calibrations"]:
        add("channel_calibration", cal["channel"], cal, cal["raw"])
    for ws in parsed["measurement_window_sets"]:
        add("measurement_window_set", str(ws["version"]), ws, ws["raw"])
    return materials


def prepare(package: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Validate, parse and hash a package; returns (parsed, materials, blob_hash)."""
    parsed = parse_package(package)
    raw_refs = {s["segment_id"]: _segment_raw(package, s["segment_id"])
                for s in parsed["segments"]}
    parsed["raw_refs"] = raw_refs
    blob_hash = digest(package)
    materials = build_materials(parsed)
    return parsed, materials, blob_hash


def _segment_raw(package: dict[str, Any], seg_id: str) -> dict[str, Any]:
    for seg in package["segments"]:
        if seg["segment_id"] == seg_id:
            return seg
    raise IngestError(f"internal: segment {seg_id} raw lookup failed")


def diagnostics(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect every non-fatal 'kept in place' diagnostic in a batch."""
    out: list[dict[str, Any]] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            if obj.get("diagnostic") and obj.get("raw") is not None:
                out.append({
                    "path": path,
                    "raw": raw_repr(obj.get("raw")),
                    "diagnostic": obj["diagnostic"],
                })
            for key, value in obj.items():
                if key in ("raw", "raw_refs"):
                    continue
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]")

    walk(parsed, "")
    return out
