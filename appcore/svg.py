"""Server-rendered SVG: overlay (叠影), jitter distribution, windows."""
from __future__ import annotations

import numpy as np

from appcore.projection import _materialized, _parse_time
from appcore import waveforms as W

WIDTH, HEIGHT = 900, 360
M = 48


def _esc(x) -> str:
    return (str(x).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def overlay_svg(case: dict, state: dict, *, show_segment: str | None = None) -> str:
    win = case["windows"] and next(
        w for w in case["windows"] if w["version"] == state["window"]["version"])
    win_lo = _parse_time(win["start"])[0]
    win_hi = _parse_time(win["end"])[0]
    segs = case["segments"]
    x0, x1 = -14e-9, 96e-9
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
             f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
             'font-family="sans-serif" font-size="11">',
             f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#0b1020"/>']
    pal = ["#7dd3fc", "#f0abfc", "#fca5a5", "#86efac", "#fde68a",
           "#fdba74", "#c4b5fd", "#67e8f9"]

    def X(t):
        return M + (t - x0) / (x1 - x0) * (WIDTH - 2 * M)

    def Y(v, vmax):
        return HEIGHT - M - (v / vmax) * (HEIGHT - 2 * M)

    # window band
    parts.append(
        f'<rect x="{X(win_lo):.1f}" y="{M}" width="{X(win_hi)-X(win_lo):.1f}" '
        f'height="{HEIGHT-2*M}" fill="#1d4ed8" opacity="0.18"/>')
    parts.append(
        f'<text x="{X(win_lo):.1f}" y="{M-6:.1f}" fill="#93c5fd">'
        f'window {_esc(win["version"])} {_esc(win["start"])}..{_esc(win["end"])}</text>')

    for k, seg in enumerate(segs):
        sid = seg["seg_id"]
        if show_segment and sid != show_segment:
            continue
        view = next((v for v in state["segments"] if v["seg_id"] == sid), None)
        color = pal[k % len(pal)]
        ch = seg["channels"]["CH1"]
        m = _materialized(ch)
        if not m["voltage_available"]:
            continue
        chosen = next((c for c in view["candidates"] if c.get("adopted"))
                      if view else None, None)
        lag = (chosen or view["candidates"][0] or {})["lag_s"] \
            if view and view["candidates"] else 0.0
        t = m["time"] - lag
        v = m["volts"]
        mask = m["mask"]
        opacity = 0.85 if show_segment else 0.55
        pts = []
        for i in range(len(t)):
            if mask[i] and np.isfinite(v[i]):
                pts.append(f"{X(t[i]):.1f},{Y(v[i], 2.0):.1f}")
        parts.append(
            f'<polyline data-uid="overlay:{_esc(sid)}:CH1" fill="none" '
            f'stroke="{color}" stroke-width="{1.6 if show_segment else 1.0}" '
            f'opacity="{opacity}" points="{" ".join(pts)}"/>')
        # gaps / rails markers
        for r in m["regions"]:
            ia, ib = r["start_index"], r["end_index"]
            rc = "#ef4444" if r["code"].startswith("saturation") else "#a16207"
            parts.append(
                f'<rect data-uid="exclude-region:{_esc(sid)}:{r["code"]}:{ia}" '
                f'x="{X(t[ia]):.1f}" y="{M}" width="4" '
                f'height="{HEIGHT-2*M}" fill="{rc}" opacity="0.5"/>')
        if show_segment:
            # trigger line
            idx = seg["channels"]["trigger"]["index"]
            if 0 <= idx < len(t):
                tx = X(t[idx])
                parts.append(
                    f'<line data-uid="trigger:{_esc(sid)}" x1="{tx:.1f}" '
                    f'x2="{tx:.1f}" y1="{M}" y2="{HEIGHT-M}" '
                    'stroke="#f472b6" stroke-dasharray="4 3"/>')
    # axes
    for gx in range(-10, 97, 10):
        parts.append(
            f'<line x1="{X(gx*1e-9):.1f}" x2="{X(gx*1e-9):.1f}" '
            f'y1="{HEIGHT-M}" y2="{HEIGHT-M+4}" stroke="#475569"/>')
        parts.append(
            f'<text x="{X(gx*1e-9)-8:.1f}" y="{HEIGHT-M+16}" fill="#94a3b8">'
            f'{gx}</text>')
    parts.append(f'<text x="{M}" y="20" fill="#e2e8f0">叠影 overlay '
                 '(CH1 volts, aligned to adopted/first candidate)</text>')
    parts.append("</svg>")
    return "".join(parts)


def jitter_svg(state: dict) -> str:
    vals = [s["jitter"]["jitter_ns"] for s in state["segments"]
            if s.get("jitter") and s["jitter"]["jitter_ns"] is not None]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
             f'height="220" viewBox="0 0 {WIDTH} 220" font-family="sans-serif" '
             'font-size="11">',
             f'<rect width="{WIDTH}" height="220" fill="#0b1020"/>']
    if not vals:
        parts.append('<text x="40" y="40" fill="#e2e8f0">no included jitter</text>')
        parts.append("</svg>")
        return "".join(parts)
    lo, hi = min(vals + [-1.0]), max(vals + [1.0])
    span = max(hi - lo, 1e-9)

    def X(v):
        return M + (v - lo) / span * (WIDTH - 2 * M)

    counts, edges = np.histogram(vals, bins=max(4, len(vals)))
    top = counts.max()
    for i, c in enumerate(counts):
        if c == 0:
            continue
        h = c / top * (140 - M)
        parts.append(
            f'<rect data-uid="jitter-bin:{i}" x="{X(edges[i]):.1f}" '
            f'y="{150-h:.1f}" width="{X(edges[i+1])-X(edges[i])-2:.1f}" '
            f'height="{h:.1f}" fill="#38bdf8" opacity="0.85"/>')
    parts.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="30" y2="150" '
                 'stroke="#f472b6" stroke-dasharray="3 3"/>')
    # per-segment ticks with stable identities
    for s in state["segments"]:
        j = s.get("jitter") or {}
        if j.get("jitter_ns") is None:
            continue
        parts.append(
            f'<circle data-uid="jitter-point:{_esc(s["seg_id"])}" '
            f'cx="{X(j["jitter_ns"]):.1f}" cy="160" r="3.5" fill="#7dd3fc"/>')
        parts.append(
            f'<text x="{X(j["jitter_ns"])-26:.1f}" y="176" fill="#94a3b8">'
            f'{_esc(s["order"])}</text>')
    parts.append('<text x="40" y="22" fill="#e2e8f0">触发抖动分布 '
                 'jitter vs mean event (ns)</text>')
    parts.append("</svg>")
    return "".join(parts)


def segment_svg(case: dict, state: dict, seg_id: str) -> str:
    seg = next(s for s in case["segments"] if s["seg_id"] == seg_id)
    win = next(w for w in case["windows"]
               if w["version"] == state["window"]["version"])
    win_lo = _parse_time(win["start"])[0]
    win_hi = _parse_time(win["end"])[0]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
             f'height="260" viewBox="0 0 {WIDTH} 260" font-family="sans-serif" '
             'font-size="11">',
             f'<rect width="{WIDTH}" height="260" fill="#0b1020"/>']
    colors = {"CH1": "#7dd3fc", "CH2": "#f0abfc"}
    xmin, xmax = 0.0, 160e-9

    def X(t):
        return M + (t - xmin) / (xmax - xmin) * (WIDTH - 2 * M)

    def Y(v):
        return 190 - M - (v / 2.0) * 120

    parts.append(
        f'<rect x="{X(win_lo):.1f}" y="30" width="{X(win_hi)-X(win_lo):.1f}" '
        f'height="150" fill="#1d4ed8" opacity="0.15"/>')
    for cid in ("CH1", "CH2"):
        m = _materialized(seg["channels"][cid])
        if not m["voltage_available"]:
            continue
        pts = []
        times, volts, mask = m["time"], m["volts"], m["mask"]
        for i in range(len(times)):
            if mask[i] and np.isfinite(volts[i]):
                pts.append(f"{X(times[i]):.1f},{Y(volts[i]):.1f}")
        parts.append(
            f'<polyline data-uid="segwave:{_esc(seg_id)}:{cid}" '
            f'fill="none" stroke="{colors[cid]}" stroke-width="1.5" '
            f'points="{" ".join(pts)}"/>')
    idx = seg["channels"]["trigger"]["index"]
    tb = seg["channels"]["CH1"]["timebase"]
    n = len(seg["channels"]["CH1"]["samples"])
    tt = W.materialize_time(tb, n).times
    if 0 <= idx < len(tt):
        parts.append(
            f'<line data-uid="segtrigger:{_esc(seg_id)}" '
            f'x1="{X(tt[idx]):.1f}" x2="{X(tt[idx]):.1f}" y1="30" y2="180" '
            'stroke="#f472b6" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="40" y="20" fill="#e2e8f0">{_esc(seg["label"])}</text>')
    parts.append('<text x="640" y="20" fill="#7dd3fc">CH1</text> '
                 '<text x="690" y="20" fill="#f0abfc">CH2 (deskew applied)</text>')
    parts.append("</svg>")
    return "".join(parts)
