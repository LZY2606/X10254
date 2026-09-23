from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "case_pack" / "waveform_overlap_station"


def pulse(t, centers, sigma=2.0, amp=0.48):
    y = np.zeros_like(np.asarray(t, dtype=float))
    for center in centers:
        y += amp * np.exp(-0.5 * ((np.asarray(t) - center) / sigma) ** 2)
    return y


def to_counts(volts, gain=0.001):
    return [int(round(v / gain)) for v in volts]


def make_segments():
    t = np.arange(0, 160, 4.0)
    s1 = pulse(t, [48, 72, 96], sigma=3.0) + 0.18*pulse(t, [64], sigma=1.2) - 0.10*pulse(t, [76], sigma=1.0)
    s2 = pulse(t, [56, 80], sigma=3.0)
    nonuniform = np.asarray([
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52,
        56, 64, 68, 72, 76, 80, 88, 92, 96, 104, 108,
    ], dtype=float)
    s3 = pulse(nonuniform, [72], sigma=3.0) + 0.18*pulse(nonuniform, [64], sigma=1.2) - 0.10*pulse(nonuniform, [76], sigma=1.0)
    s4 = pulse(t, [48, 72, 96], sigma=3.0) + 0.18*pulse(t, [64], sigma=1.2) - 0.10*pulse(t, [76], sigma=1.0)
    s4_counts = to_counts(s4)
    # Missing samples are explicit nulls. Saturation is a physical interval recorded
    # with raw codes, and remains visible but excluded from interpolation/correlation.
    for index in [1, 2, 30, 31]:
        if index < len(s4_counts):
            s4_counts[index] = None
    for index in range(17, 19):
        if index < len(s4_counts) and s4_counts[index] is not None:
            s4_counts[index] = 507
    s4_points = [{"value": float(x), "unit": "ns"} for x in t]
    return [
        {
            "segment_id": "S1",
            "label": "reference periodic burst",
            "timebase": {"type": "uniform", "origin": {"value": 0, "unit": "ns"},
                         "increment": {"value": 4, "unit": "ns"}, "units": "seconds"},
            "trigger": {"timestamp": {"value": 70, "unit": "ns"}, "source": "hardware_trigger_line",
                        "coordinate_basis": "segment_local_time_before_channel_deskew"},
            "channels": {
                "CH1": {"raw_samples": to_counts(s1), "saturated_intervals": []},
                "CH2": {"raw_samples": to_counts(s1, 0.002), "saturated_intervals": []},
            },
        },
        {
            "segment_id": "S2",
            "label": "double legal cross-correlation ambiguity",
            "timebase": {"type": "uniform", "origin": {"value": 0, "unit": "ns"},
                         "increment": {"value": 4, "unit": "ns"}, "units": "seconds"},
            "trigger": {"timestamp": {"value": 78, "unit": "ns"}, "source": "hardware_trigger_line",
                        "coordinate_basis": "segment_local_time_before_channel_deskew"},
            "channels": {
                "CH1": {"raw_samples": to_counts(s2), "saturated_intervals": []},
                "CH2": {"raw_samples": to_counts(s2, 0.002), "saturated_intervals": []},
            },
        },
        {
            "segment_id": "S3",
            "label": "unequal sample intervals and channel deskew",
            "timebase": {"type": "non_uniform", "points_seconds": s4_points[:0] +
                         [{"value": float(x), "unit": "ns"} for x in nonuniform],
                         "units": "seconds", "interval_rule": "actual_timestamps_no_index_subtraction"},
            "trigger": {"timestamp": {"value": 92, "unit": "ns"}, "source": "hardware_trigger_line",
                        "coordinate_basis": "segment_local_time_before_channel_deskew"},
            "channels": {
                "CH1": {"raw_samples": to_counts(s3), "saturated_intervals": []},
                # Raw event appears at 102 ns; 10 ns deskew maps it to 92 ns coordinate.
                "CH2": {"raw_samples": to_counts(pulse(nonuniform, [102]), gain=0.002), "saturated_intervals": []},
            },
        },
        {
            "segment_id": "S4",
            "label": "missing samples, saturation, and boundary trigger",
            "timebase": {"type": "uniform", "origin": {"value": 0, "unit": "ns"},
                         "increment": {"value": 4, "unit": "ns"}, "units": "seconds",
                         "boundary_note": "capture start is a legal boundary trigger at 0 ns"},
            "trigger": {"timestamp": {"value": 0, "unit": "ns"}, "source": "hardware_trigger_line",
                        "coordinate_basis": "segment_local_time_before_channel_deskew",
                        "boundary_position": "capture_start"},
            "channels": {
                "CH1": {"raw_samples": s4_counts,
                        "saturated_intervals": [{"start": {"value": 66, "unit": "ns"},
                                                 "end": {"value": 71, "unit": "ns"},
                                                 "reason": "ADC rail overrange"}]},
                "CH2": {"raw_samples": to_counts(s4, 0.002), "saturated_intervals": []},
            },
        },
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pack = {
        "case_pack_id": "waveform-overlap-station-v1",
        "title": "波形叠影台：示波器分段、触发抖动与测量窗口谱系",
        "domain": "oscilloscope_segmented_capture_trigger_jitter_measurement_window",
        "schema_version": "1.0",
        "narrative_stage": "initial_import",
        "channels": {
            "CH1": {
                "channel_id": "CH1",
                "vertical_gain": {"value": 1.0, "unit": "mV/count", "quantity": "vertical_gain"},
                "probe_attenuation": {"display": "X10", "ratio_ohm_field_preserved": "X10"},
                "vertical_offset": {"value": 0, "unit": "mV", "coordinate_basis": "probe_tip_voltage"},
                "deskew": {"value": 0, "unit": "ns", "coordinate_basis": "channel_delay_to_reference"},
                "coupling": "DC_1M",
            },
            "CH2": {
                "channel_id": "CH2",
                "vertical_gain": {"value": 2.0, "unit": "mV/count", "quantity": "vertical_gain"},
                "probe_attenuation": {"display": "1:1"},
                "vertical_offset": {"value": "not-a-calibrated-offset", "unit": "mV",
                                    "diagnostic_expected": "OFFSET_WITHOUT_NUMERIC_VALUE"},
                "deskew": {"value": 10, "unit": "ns", "coordinate_basis": "channel_delay_to_reference"},
                "coupling": "DC_50",
            },
        },
        "segments": make_segments(),
        "measurement_window_versions": {
            "W-MAIN": [
                {"window_id": "W-MAIN", "version": 1,
                 "start": {"value": 35, "unit": "ns"}, "end": {"value": 95, "unit": "ns"},
                 "coordinate_basis": "aligned_time", "status": "active",
                 "rule": "missing and saturated samples are excluded rather than interpolated"},
            ]
        },
        "alignment_rules": {
            "ALIGN-REF": {
                "rule_id": "ALIGN-REF",
                "version": 1,
                "reference_segment": "S1",
                "reference_channel": "CH1",
                "candidate_threshold": 0.40,
                "method": "time-coordinate cross-correlation; preserve every legal peak",
                "interpolation": "forbidden on missing or saturated intervals",
            }
        },
    }
    (OUT / "manifest.json").write_text(json.dumps({
        "case_pack_id": pack["case_pack_id"],
        "title": pack["title"],
        "files": ["01_initial_import.json", "02_replay_events.json", "counterexamples.json"],
    }, ensure_ascii=False, indent=2) + "\n")
    (OUT / "01_initial_import.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n")
    replay = [
        {"event_index": 1, "kind": "parameter_revision", "rule_id": "ALIGN-REF", "version": 2,
         "patch": {"candidate_threshold": 0.70}, "reason": "tighten multi-peak acceptance threshold"},
        {"event_index": 2, "kind": "measurement_window_revision", "window_id": "W-MAIN", "version": 2,
         "start": {"value": 40, "unit": "ns"}, "end": {"value": 100, "unit": "ns"},
         "reason": "move derived measurement interval while retaining version 1"},
        {"event_index": 3, "kind": "source_data_revision", "scope": "local_dependency",
         "channel_id": "CH2", "deskew": {"value": 12, "unit": "ns"},
         "reason": "probe cable delay recalibrated; only CH2-derived values are invalidated"},
        {"event_index": 4, "kind": "conflicting_manual_decisions", "reference_event": 2,
         "decisions": [
             {"branch_id": "BR-ACCEPT-LAG-8", "segment_id": "S2",
              "candidate_id": "cand:lag:8e-09s", "decision": "accept",
              "reason": "reviewer selects the earlier repeat"},
             {"branch_id": "BR-ACCEPT-LAG-NEG-8", "segment_id": "S2",
              "candidate_id": "cand:lag:-1.6e-08s", "decision": "accept",
              "reason": "second reviewer selects the other legal repeat and conflicts with branch one"},
         ]},
    ]
    (OUT / "02_replay_events.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n")
    counterexamples = [
        {
            "counterexample_id": "CX-01-UNEQUAL-TIMEBASE", "direct_source_changed": "timebase.points_seconds",
            "baseline_event": "01_initial_import", "changed_segment": "S3",
            "mutation": "replace uniform 4 ns assumption with actual non-uniform timestamps",
            "adopted_result": {"valid": True, "basis": "SHIFT-CANDIDATES-USE-ACTUAL-TIME"},
            "not_adopted_result": {"valid": False, "basis": "INDEX-DIFFERENCE-AS-TIME"},
            "machine_judgment": {"event_type": "derivation_rule_judgment",
                                 "rule": "time_unequal_no_index_subtraction", "decision": "adopt_time_coordinates"}},
        {
            "counterexample_id": "CX-02-VERTICAL-SATURATION", "direct_source_changed": "CH1.saturated_intervals",
            "baseline_event": "01_initial_import", "changed_segment": "S4",
            "mutation": "mark rail overrange from 68 ns to 80 ns",
            "adopted_result": {"valid": True, "basis": "MASK-SATURATED-SAMPLES"},
            "not_adopted_result": {"valid": False, "basis": "INTERPOLATE-ACROSS-SATURATION"},
            "machine_judgment": {"event_type": "derivation_rule_judgment",
                                 "rule": "saturated_intervals_no_interpolation", "decision": "exclude_interval"}},
        {
            "counterexample_id": "CX-03-MULTIPEAK-ALIGNMENT", "direct_source_changed": "segment waveform pattern",
            "baseline_event": "01_initial_import", "changed_segment": "S2",
            "mutation": "two repeated events both satisfy threshold",
            "adopted_result": {"valid": True, "basis": "PRESERVE-LEGAL-LAG-8-AND-NEG8"},
            "not_adopted_result": {"valid": False, "basis": "KEEP-MAXIMUM-ONLY"},
            "machine_judgment": {"event_type": "derivation_rule_judgment",
                                 "rule": "preserve_all_cross_correlation_peaks", "decision": "retain_ambiguous_candidates"}},
        {
            "counterexample_id": "CX-04-CHANNEL-DESKEW", "direct_source_changed": "CH2.deskew",
            "baseline_event": "01_initial_import", "changed_segment": "S3",
            "mutation": "change only CH2 delay by 2 ns",
            "adopted_result": {"valid": True, "basis": "INVALIDATE-CH2-ONLY"},
            "not_adopted_result": {"valid": False, "basis": "GLOBAL-REALIGN-ALL-CHANNELS"},
            "machine_judgment": {"event_type": "dependency_invalidation_judgment",
                                 "rule": "local_dependency_invalidation", "decision": "limit_scope_to_channel"}},
        {
            "counterexample_id": "CX-05-MISSING-SAMPLES", "direct_source_changed": "CH1.raw_samples",
            "baseline_event": "01_initial_import", "changed_segment": "S4",
            "mutation": "null raw samples at capture tail and near trigger",
            "adopted_result": {"valid": True, "basis": "NULLS-REMAIN-AND-ARE-MASKED"},
            "not_adopted_result": {"valid": False, "basis": "FILL-MISSING-FOR-PLOT"},
            "machine_judgment": {"event_type": "derivation_rule_judgment",
                                 "rule": "missing_samples_no_interpolation", "decision": "keep_original_nulls"}},
        {
            "counterexample_id": "CX-06-TRIGGER-AT-BOUNDARY", "direct_source_changed": "trigger.timestamp",
            "baseline_event": "01_initial_import", "changed_segment": "S4",
            "mutation": "trigger at 0 s capture_start",
            "adopted_result": {"valid": True, "basis": "BOUNDARY-TRIGGER-DIAGNOSTIC"},
            "not_adopted_result": {"valid": False, "basis": "DROP-EDGE-TRIGGER"},
            "machine_judgment": {"event_type": "trigger_boundary_judgment",
                                 "rule": "boundary_trigger_is_legal", "decision": "retain_with_diagnostic"}},
        {
            "counterexample_id": "CX-07-WINDOW-VERSION", "direct_source_changed": "measurement window W-MAIN",
            "baseline_event": "01_initial_import", "changed_segment": "ALL",
            "mutation": "revise interval from v1 to v2 without deleting v1",
            "adopted_result": {"valid": True, "basis": "VERSIONED-WINDOW-DEPENDENCIES"},
            "not_adopted_result": {"valid": False, "basis": "OVERWRITE-WINDOW-GLOBALLY"},
            "machine_judgment": {"event_type": "dependency_invalidation_judgment",
                                 "rule": "measurement_window_versioned", "decision": "retain_old_branch_state"}},
    ]
    (OUT / "counterexamples.json").write_text(json.dumps(counterexamples, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
