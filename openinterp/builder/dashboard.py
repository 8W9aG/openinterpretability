"""Static SVG geometry for the home page's registry dashboard, computed at build time.

The dashboard has three charts of experiment verdicts:

- the timeline: experiments per day, stacked by verdict, and the share of experiments
  up to that day that were VERIFIED;
- verdicts per model, as one stacked bar for each model;
- verdicts per layer, as stacked columns, one small chart for each model.

The input is the rows of the ``verdicts`` query in
:mod:`openinterp.builder.registry_stats`: one row for each (day, model, layer, verdict)
with a count. Dictionary features are not in it, because they are a baseline and not
claims about native units. Each function returns plain coordinates and labels; the
``_dashboard.html`` template draws them.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, timedelta

from openinterp.builder.charts import _nice_step
from openinterp.builder.statuses import MAP_STATUSES
from openinterp.core.schemas import VerdictStatus

# The verdicts an experiment can get, in the site's status order. VERIFIED is the bottom
# segment of each stack, so its height is easy to compare.
VERDICTS: tuple[str, ...] = tuple(s for s in MAP_STATUSES if s in VerdictStatus.__args__)
DAILY_BINS = 120  # above this many days, the timeline bins by week (Monday to Sunday)
TIME_LABELS = 6  # x-axis labels on the timeline, at most


def dashboard(rows: list[dict], models) -> dict | None:
    """All three charts of the dashboard, or ``None`` when there are no verdicts."""
    if not rows:
        return None
    return {
        "statuses": VERDICTS,
        "timeline": timeline(rows),
        "models": model_bars(rows, models),
        "layers": layer_columns(rows, models),
    }


def _counts_line(counts: Counter) -> str:
    """For example ``12 VERIFIED, 3 NEEDS REVISION``, in status order, zeros left out."""
    return ", ".join(f"{counts[s]} {s.replace('_', ' ')}" for s in VERDICTS if counts[s])


def _stack(counts: Counter, x: float, w: float, y0: float, scale: float) -> list[dict]:
    """Column segments from the baseline ``y0`` up, VERIFIED at the bottom."""
    segments, y = [], y0
    for s in VERDICTS:
        if counts[s]:
            h = counts[s] * scale
            y -= h
            segments.append({"status": s, "x": x, "y": round(y, 2), "w": w, "h": round(h, 2)})
    return segments


def _bins(rows: list[dict]) -> list[tuple[date, Counter]]:
    """Verdict counts per day, or per week when the span is long. Empty days are kept."""
    by_day: dict[date, Counter] = defaultdict(Counter)
    for r in rows:
        by_day[date.fromisoformat(r["day"])][r["status"]] += r["n"]
    first, last = min(by_day), max(by_day)
    step = 1
    if (last - first).days + 1 > DAILY_BINS:
        step = 7
        first -= timedelta(days=first.weekday())
    bins = []
    start = first
    while start <= last:
        counts: Counter = Counter()
        for i in range(step):
            counts.update(by_day.get(start + timedelta(days=i), Counter()))
        bins.append((start, counts))
        start += timedelta(days=step)
    return bins


def _column(counts: Counter, i: int, pitch: float, left: float, base: float, scale: float):
    """Slot ``i`` of a column chart: a hover area as wide as the slot, and its stack."""
    col_w = round(max(pitch - 2, 1.0), 2)
    x = round(left + i * pitch + (pitch - col_w) / 2, 2)
    n = sum(counts.values())
    return {
        "hover_x": round(left + i * pitch, 2),
        "segments": _stack(counts, x, col_w, base, scale),
        "n": n,
        "counts": counts,
        "share": counts["VERIFIED"] / n if n else None,
    }


def _plural(n: int) -> str:
    return f"{n} experiment{'s' if n != 1 else ''}"


def timeline(rows: list[dict], width: int = 640) -> dict:
    """Two panels on one time axis: experiments per bin, and the VERIFIED share to date."""
    bins = _bins(rows)
    weekly = len(bins) > 1 and (bins[1][0] - bins[0][0]).days == 7
    left, right = 40, width - 8
    count_top, count_h, gap, share_h = 6, 120, 26, 70
    pitch = (right - left) / len(bins)
    peak = max(sum(c.values()) for _, c in bins) or 1
    step = max(1, int(_nice_step(peak)))  # counts are whole numbers
    reach = math.ceil(peak / step) * step  # the next tick at or above the peak
    base, scale = count_top + count_h, count_h / reach
    share_base = base + gap + share_h
    columns = _timeline_columns(bins, weekly, pitch, left, base, scale)
    points = [
        (round(c["hover_x"] + pitch / 2, 2), round(share_base - c["to_date"] * share_h, 2))
        for c in columns
        if c["to_date"] is not None
    ]
    every = max(1, -(-len(bins) // TIME_LABELS))
    return {
        "width": width,
        "height": share_base + 18,
        "left": left,
        "right": right,
        "pitch": round(pitch, 2),
        "count_top": count_top,
        "count_base": base,
        "share_base": share_base,
        "count_ticks": [(round(base - v * scale, 2), f"{v:g}") for v in range(0, reach + 1, step)],
        "share_ticks": [(round(share_base - f * share_h, 2), f"{f:.0%}") for f in (0, 0.5, 1)],
        "columns": columns,
        "line": " ".join(f"{'M' if i == 0 else 'L'}{x} {y}" for i, (x, y) in enumerate(points)),
        "last_point": points[-1] if points else None,
        "share": columns[-1]["to_date"],
        "x_labels": [
            (round(left + i * pitch + pitch / 2, 2), f"{start:%b} {start.day}")
            for i, (start, _) in enumerate(bins)
            if i % every == 0
        ],
        "unit": "week" if weekly else "day",
    }


def _timeline_columns(bins, weekly: bool, pitch, left, base, scale) -> list[dict]:
    """One column per bin, with the VERIFIED share of all experiments up to its end."""
    columns = []
    done = verified = 0
    for i, (start, counts) in enumerate(bins):
        col = _column(counts, i, pitch, left, base, scale)
        done += col["n"]
        verified += counts["VERIFIED"]
        to_date = verified / done if done else None
        label = f"{'week of ' if weekly else ''}{start:%Y-%m-%d} · {_plural(col['n'])}"
        if col["n"]:
            label += f": {_counts_line(counts)}"
        if to_date is not None:
            label += f"\nVERIFIED to date: {to_date:.0%} ({verified} of {done})"
        columns.append({**col, "date": start.isoformat(), "to_date": to_date, "label": label})
    return columns


def _ordered_models(rows: list[dict], models) -> list[str]:
    """Model names in the order of the home page's model list, then any others by name."""
    present = {r["model_name"] for r in rows}
    order = [m.name for m in models if m.name in present]
    return order + sorted(present - set(order))


def model_bars(rows: list[dict], models, width: int = 640) -> dict:
    """One horizontal bar per model, its length the experiment count, stacked by verdict."""
    totals: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        totals[r["model_name"]][r["status"]] += r["n"]
    names = _ordered_models(rows, models)
    left, right = 168, width - 48
    bar, pitch, top = 16, 26, 4
    peak = max(sum(c.values()) for c in totals.values()) or 1
    scale = (right - left) / peak
    out = []
    for i, name in enumerate(names):
        counts = totals[name]
        n = sum(counts.values())
        y = top + i * pitch
        segments, x = [], float(left)
        for s in VERDICTS:
            if counts[s]:
                w = counts[s] * scale
                segments.append({"status": s, "x": round(x, 2), "y": y, "w": round(w, 2), "h": bar})
                x += w
        out.append(
            {
                "name": name,
                "y": y,
                "n": n,
                "end_x": round(x + 6, 2),
                "segments": segments,
                "counts": counts,
                "share": counts["VERIFIED"] / n,
                "label": f"{name} · {n} experiments: {_counts_line(counts)}",
            }
        )
    return {
        "width": width,
        "height": top + len(names) * pitch,
        "left": left,
        "bar": bar,
        "rows": out,
    }


def layer_columns(rows: list[dict], models, width: int = 360) -> list[dict]:
    """One small chart per model: a column of verdicts for each layer, on the model's scale.

    Every layer of the model gets a slot, studied or not, so gaps in coverage show.
    """
    by_layer: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        by_layer[r["model_name"]][r["layer"]][r["status"]] += r["n"]
    n_layers = {m.name: m.n_layers for m in models}
    return [
        _layer_chart(name, by_layer[name], n_layers.get(name, 0), width)
        for name in _ordered_models(rows, models)
    ]


def _layer_chart(name: str, layers: dict[int, Counter], n_layers: int, width: int) -> dict:
    """The verdict columns of one model's layers."""
    left, right, top, height = 32, width - 4, 6, 90
    base = top + height
    count = max(n_layers, max(layers) + 1)
    pitch = (right - left) / count
    peak = max(sum(c.values()) for c in layers.values()) or 1
    columns = []
    for layer in range(count):
        col = _column(layers.get(layer, Counter()), layer, pitch, left, base, height / peak)
        label = f"{name} L{layer} · {_plural(col['n'])}"
        if col["n"]:
            label += f": {_counts_line(col['counts'])} ({col['share']:.0%} VERIFIED)"
        columns.append({**col, "layer": layer, "label": label})
    every = 1 if count <= 12 else 2 if count <= 24 else 4
    return {
        "name": name,
        "width": width,
        "height": base + 16,
        "left": left,
        "right": right,
        "top": top,
        "base": base,
        "pitch": round(pitch, 2),
        "peak": peak,
        "columns": columns,
        "x_labels": [
            (round(left + i * pitch + pitch / 2, 2), f"L{i}") for i in range(0, count, every)
        ],
    }
