"""派生层：对齐重复事件、抖动分布、测量窗口与排除判定。

三条硬性规则（对应机器可读事件代码）：
  R-NONUNIFORM : 时间轴不等间隔时不用索引相减，一律用真实时间坐标；
  R-NOCLEAN    : 缺样与饱和区间不进入插值/对齐，只做掩码；
  R-MULTIPEAK  : 对齐保留多峰互相关候选，不唯一时保留全部合法候选。

派生结果不可变：参数/源数据/窗口版本变化后只产生新版本对象，
旧对象由调用方（store）保留并据此计算失效范围，绝不全局替换。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ids import stable_id

GRID_DT = 1.25e-9           # 对齐分析栅格（不改变原始采样，只用于派生）
HALF_WIN = 45e-9            # 触发局部窗口半宽（秒）
MIN_COVERAGE = 0.80         # 栅格覆盖率下限（处理触发在边界）


@dataclass(frozen=True)
class SegmentData:
    seg_id: str
    order: int
    channel_id: str
    counts: list[int | None]
    flags: list[str]          # ok/missing/saturated
    dt: object                # 数字（秒）或 None（走显式时间）
    dt_basis: str
    explicit_times: object    # 显式时间（相对触发，秒）或 None
    trigger_index: int
    trigger_at_boundary: bool
    vgain_si: float
    vgain_raw: str
    atten: float
    atten_raw: str
    deskew_si: float
    deskew_raw: str
    trigger_ts_raw: str
    trigger_ts_si: float
    nonuniform: bool
    nonuniformity: float
    modified: bool            # 源数据已被局部修订（不进参考模板）


@dataclass(frozen=True)
class RuleParams:
    min_peak: float
    peak_min_sep_s: float
    grid_dt: float
    half_win_s: float
    min_coverage: float
    params_version: str
    raw: dict[str, str]


@dataclass(frozen=True)
class WindowVersion:
    window_version: str
    start_raw: str
    end_raw: str
    start_s: float | None
    end_s: float | None
    scale_basis: str | None
    unconvertible: bool
    diagnostic: str | None


def physical_voltage(seg: SegmentData) -> np.ndarray:
    """counts -> 探头端电压（V 基准），缺样为 nan。"""
    vals = np.array([np.nan if c is None else c for c in seg.counts],
                    dtype=np.float64)
    return vals * seg.vgain_si * seg.atten


def sample_times(seg: SegmentData) -> np.ndarray:
    """每点时间（相对触发，秒）。不等间隔走显式时间，绝不索引相减。"""
    if seg.explicit_times is not None:
        return np.array(seg.explicit_times, dtype=np.float64) - seg.deskew_si
    return ((np.arange(len(seg.counts)) - seg.trigger_index)
            * float(seg.dt) - seg.deskew_si)


def resample_on_grid(times: np.ndarray, volts: np.ndarray, flags: list[str],
                     grid: np.ndarray) -> np.ndarray:
    """把有效（非缺样、非饱和）锚点线性映射到分析栅格。

    缺样/饱和点不做插值：栅格点两侧若夹着被排除锚点，则该栅格点置 nan。
    """
    valid = np.array([f == "ok" for f in flags], dtype=bool)
    ta = times[valid]
    va = volts[valid]
    order = np.argsort(ta)
    ta, va = ta[order], va[order]
    out = np.full_like(grid, np.nan)
    for i, tg in enumerate(grid):
        if tg < ta[0] or tg > ta[-1]:
            continue
        j = int(np.searchsorted(ta, tg))
        if j < len(ta) and ta[j] == tg:
            out[i] = va[j]
            continue
        lo, hi = j - 1, j
        if lo < 0 or hi >= len(ta):
            continue
        # 间隙内部若原本存在被排除锚点，则不插值（缺样/饱和不进入插值）
        mid = (times > ta[lo]) & (times < ta[hi])
        if mid.any() and not valid[mid].all():
            continue
        frac = (tg - ta[lo]) / (ta[hi] - ta[lo])
        out[i] = va[lo] + frac * (va[hi] - va[lo])
    return out


# ---------------------------------------------------------------------------
# 对齐：参考模板（有效锚点中值）+ 多峰归一化互相关
# ---------------------------------------------------------------------------

def _grid(params: RuleParams) -> np.ndarray:
    half = params.half_win_s
    step = params.grid_dt
    n = int(round(half / step))
    return np.arange(-n, n + 1) * step


def _build_template(segments: list[SegmentData], grid: np.ndarray,
                    params: RuleParams) -> tuple[np.ndarray | None, list[str]]:
    """用原始、A 通道、无缺样/饱和、未修订的段在触发坐标上取中值作模板。"""
    stack = []
    used: list[str] = []
    for seg in segments:
        if seg.modified or seg.channel_id != "A" or any(f != "ok" for f in seg.flags):
            continue
        g = resample_on_grid(sample_times(seg), physical_voltage(seg), seg.flags, grid)
        if np.isfinite(g).mean() >= params.min_coverage:
            stack.append(g)
            used.append(seg.seg_id)
    if not stack:
        return None, []
    return np.nanmedian(np.vstack(stack), axis=0), used


def _norm_xcorr(template: np.ndarray, trace: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """有效栅格掩码下的归一化互相关；缺样/饱和栅格不参与。"""
    t = np.where(mask, template, 0.0)
    x = np.where(mask, trace, 0.0)
    t = t - t.mean()
    x = x - x.mean()
    denom = np.sqrt(np.sum(mask) * np.sum(t * t) * np.sum(x * x))
    if denom <= 0:
        return np.full(2 * len(trace) - 1, -np.inf)
    return np.correlate(t, x, mode="full") / denom


def _peaks(score: np.ndarray, lags: np.ndarray, params: RuleParams
           ) -> list[tuple[float, float]]:
    """局部极大值 -> 按阈值与最小峰间距（时间单位）保留多个峰。"""
    cands: list[tuple[float, float]] = []
    for i in range(1, len(score) - 1):
        if (score[i] >= score[i - 1] and score[i] >= score[i + 1]
                and score[i] >= params.min_peak):
            cands.append((float(score[i]), float(lags[i])))
    cands.sort(reverse=True)
    kept: list[tuple[float, float]] = []
    for score_val, lag in cands:
        if all(abs(lag - k[1]) * params.grid_dt >= params.peak_min_sep_s
               for k in kept):
            kept.append((score_val, lag))
    return kept
