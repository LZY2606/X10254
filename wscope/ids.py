"""稳定身份：所有身份来自内容哈希，不含时间戳/随机数。

导出 -> 删除本地数据库 -> 再导入后，身份、顺序与来源关系必须保持，
因此 id 只能由材料内容与分支内序号决定。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def content_hash(obj: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical(obj)).hexdigest()
    return digest[:length] if length else digest


def stable_id(prefix: str, *parts: Any, length: int = 12) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            h.update(bytes(part))
        else:
            h.update(canonical(part))
        h.update(b"|")
    return f"{prefix}_{h.hexdigest()[:length]}"
