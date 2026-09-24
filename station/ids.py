"""内容寻址的稳定身份与规范化 JSON。

身份只依赖领域内容（原始字节、依赖、规则版本），因此导出、删除本地
数据库再导入后，material / result / candidate / event 身份保持不变。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _to_native(obj: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None  # type: ignore
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _to_native(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def content_hash(prefix: str, obj: Any) -> str:
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def material_id(payload: dict) -> str:
    return content_hash(
        "mat",
        {
            "segment_no": payload.get("segment_no"),
            "kind": payload["kind"],
            "channel": payload.get("channel"),
            "version": payload.get("version"),
            "raw_repr": payload["raw_repr"],
            "unit": payload.get("unit"),
            "coordinate_basis": payload.get("coordinate_basis"),
        },
    )


def event_id(project_id: str, branch_id: str, seq: int, prev: str | None,
             etype: str, payload: dict) -> str:
    return content_hash(
        "evt",
        {
            "project_id": project_id,
            "branch_id": branch_id,
            "seq": seq,
            "prev_event_id": prev,
            "type": etype,
            "payload": payload,
        },
    )


def result_id(kind: str, subject: str, rule_version: str, deps: list, body: dict) -> str:
    from station import DERIVE_VERSION

    return content_hash(
        "res",
        {
            "derive": DERIVE_VERSION,
            "kind": kind,
            "subject": subject,
            "rule_version": rule_version,
            "deps": deps,
            "body": body,
        },
    )


def candidate_id(subject: str, lag_ns: float, score: float, rule_version: str) -> str:
    return content_hash(
        "cand",
        {
            "subject": subject,
            "lag_ns": round(float(lag_ns), 6),
            "score": round(float(score), 6),
            "rule_version": rule_version,
        },
    )


def branch_id(project_id: str, name: str, parent_branch_id: str | None,
              root_event_id: str | None) -> str:
    return content_hash(
        "br",
        {
            "project_id": project_id,
            "name": name,
            "parent_branch_id": parent_branch_id,
            "root_event_id": root_event_id,
        },
    )
