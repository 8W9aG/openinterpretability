"""Static SVG chart geometry for experiment pages, computed at build time.

Each function returns plain coordinates and labels; the templates draw them.
"""

from __future__ import annotations

import math
from functools import partial

from openinterp.builder.statuses import MAP_STATUSES
from openinterp.core.schemas import Histogram


def histogram_bars(hist: Histogram | None, width: int = 600, height: int = 140) -> list[dict]:
    """SVG bar geometry; heights are log-scaled because activations are heavy-tailed."""
    if hist is None or not hist.counts:
        return []
    peak = math.log1p(max(hist.counts)) or 1.0
    bar_w = width / len(hist.counts)
    bars = []
    for i, count in enumerate(hist.counts):
        h = height * math.log1p(count) / peak
        bars.append(
            {
                "x": round(i * bar_w, 2),
                "y": round(height - h, 2),
                "w": round(max(bar_w - 1, 0.5), 2),
                "h": round(h, 2),
                "label": f"[{hist.bin_edges[i]:.3g}, {hist.bin_edges[i + 1]:.3g}): {count}",
            }
        )
    return bars


PROBE_ROWS = (
    ("hard_negative", "Hard negatives", "stay below"),
    ("positive_holdout", "Positive holdouts", "reach it"),
)


def probe_chart(verdict, scale: float | None, width: int = 640) -> dict | None:
    """Strip-chart geometry: each activation probe as a dot on one pole-strength axis.

    The reviewer's threshold (``activation_threshold × max activation``) is the line a
    hard negative must stay below and a holdout must reach. Dots that would overlap are
    dodged vertically within their row, so every probe stays visible.
    """
    if verdict is None or verdict.policy is None or not scale:
        return None
    kinds = {kind for kind, *_ in PROBE_ROWS}
    tests = [
        t for t in verdict.adversarial_test_results if t.kind in kinds and t.activation is not None
    ]
    if not tests:
        return None
    threshold = verdict.policy.activation_threshold * scale
    lo = min(0.0, *(t.activation for t in tests))
    hi = max(scale, *(t.activation for t in tests))
    left, right = 132, width - 12
    r, row_gap, top = 5, 64, 34
    x = partial(_axis_x, lo=lo, hi=hi, left=left, right=right)
    rows = []
    for i, (kind, label, expect) in enumerate(PROBE_ROWS):
        cy = top + 24 + i * row_gap
        group = sorted((t for t in tests if t.kind == kind), key=lambda t: t.activation)
        dots = _probe_dots(group, x, cy, r, scale)
        passed = sum(d["passed"] for d in dots)
        row = {"label": label, "expect": expect, "y": cy, "dots": dots}
        rows.append({**row, "passed": passed, "n": len(dots)})
    axis_y = top + 24 + len(PROBE_ROWS) * row_gap - 24
    return {
        "width": width,
        "height": axis_y + 20,
        "left": left,
        "right": right,
        "top": top,
        "axis_y": axis_y,
        "rows": rows,
        "r": r,
        "threshold": threshold,
        "threshold_x": x(threshold),
        "threshold_frac": verdict.policy.activation_threshold,
        "zero_x": x(0.0),
        "max_x": x(scale),
        "scale": scale,
    }


def _axis_x(v: float, *, lo: float, hi: float, left: float, right: float) -> float:
    return round(left + (right - left) * (v - lo) / ((hi - lo) or 1.0), 2)


def _probe_dots(group, x, cy: float, r: float, scale: float) -> list[dict]:
    """One row's dots, each dodged vertically off the dots already placed."""
    placed: list[tuple[float, float]] = []
    dots = []
    for t in group:
        cx = x(t.activation)
        for dy in (0, -10, 10, -20, 20):
            if all((cx - px) ** 2 + (cy + dy - py) ** 2 >= (2 * r + 1) ** 2 for px, py in placed):
                break
        placed.append((cx, cy + dy))
        prompt = t.prompt if len(t.prompt) <= 160 else t.prompt[:157] + "…"
        dots.append(
            {
                "cx": cx,
                "cy": cy + dy,
                "passed": t.passed,
                "label": f"{'passed' if t.passed else 'failed'} · {t.activation:.3f} "
                f"({t.activation / scale:.0%} of max)\n{prompt}",
            }
        )
    return dots


LENSES = (
    (
        "jacobian",
        "Jacobian lens",
        "Output direction transported to the final layer with the "
        "context-averaged Jacobian (Gurnee et al., 2026)",
    ),
    ("logit", "Logit lens", "Output direction read directly through the unembedding (J = I)"),
)
VISIBLE_WHITESPACE = str.maketrans({" ": "␣", "\n": "↵", "\t": "⇥"})


def _nice_step(span: float, max_ticks: int = 4) -> float:
    """The smallest 1/2/5 × 10ⁿ step that splits ``span`` into at most ``max_ticks``."""
    if span <= 0:
        return 1.0
    base = 10 ** math.floor(math.log10(span / max_ticks))
    return next(m * base for m in (1, 2, 5, 10) if span / (m * base) <= max_ticks)


def _bar_path(x0: float, x1: float, y: float, h: float, r: float = 4) -> str:
    """A bar from the baseline ``x0`` to ``x1``: square at the baseline, rounded at the end."""
    sign = 1 if x1 >= x0 else -1
    r = min(r, abs(x1 - x0), h / 2)
    tip = x1 - sign * r
    return (
        f"M{x0:.2f} {y:.2f}H{tip:.2f}Q{x1:.2f} {y:.2f} {x1:.2f} {y + r:.2f}"
        f"V{y + h - r:.2f}Q{x1:.2f} {y + h:.2f} {tip:.2f} {y + h:.2f}H{x0:.2f}Z"
    )


def vocab_chart(lens, width: int = 360) -> dict | None:
    """Diverging bars of each lens's top promoted and suppressed tokens by vocabulary z-score.

    One panel per lens, on a shared axis (z-scores are scale-invariant, so the lenses are
    comparable). Rows run from the most promoted token down to the most suppressed; each
    token's label sits on the far side of zero from its bar. Records without z-scores get
    no chart; the page lists their tokens instead.
    """
    lists = _vocab_lists(lens)
    if lists is None:
        return None
    zmax = max((abs(v) for pairs in lists.values() for _, v in pairs), default=0.0) or 1.0
    step = _nice_step(zmax)
    reach = math.ceil(zmax / step) * step
    zero, half = width / 2, width / 2 - 6
    bar, pitch, gap, top = 11, 15, 10, 4

    x = partial(_zero_x, zero=zero, half=half, reach=reach)
    panels = []
    for key, title, hint in LENSES:
        sides = (
            ("promoted", lists[key, "promoted"]),
            ("suppressed", lists[key, "suppressed"][::-1]),
        )
        rows = _vocab_rows(sides, x, zero, top, bar, pitch, gap)
        panels.append({"key": key, "title": title, "hint": hint, "rows": rows})
    axis_y = max((len(p["rows"]) for p in panels), default=0) * pitch + top + gap
    ticks = [
        (x(i * step), f"{i * step:+g}" if i else "0")
        for i in range(-round(reach / step), round(reach / step) + 1)
    ]
    return {
        "width": width,
        "height": axis_y + 18,
        "bar": bar,
        "pitch": pitch,
        "zero_x": zero,
        "axis_y": axis_y,
        "ticks": ticks,
        "panels": panels,
        "table": _vocab_table(lists),
    }


def _vocab_lists(lens) -> dict | None:
    """(lens, side) -> [(token, z)], or ``None`` when a list has no z-scores."""
    if lens is None:
        return None
    lists = {}
    for key, *_ in LENSES:
        for side in ("promoted", "suppressed"):
            tokens, z = getattr(lens, f"{key}_{side}"), getattr(lens, f"{key}_{side}_z")
            if z is None or len(z) != len(tokens):
                return None
            lists[key, side] = list(zip(tokens, z, strict=True))
    return lists


def _zero_x(v: float, *, zero: float, half: float, reach: float) -> float:
    return round(zero + half * v / reach, 2)


def _vocab_rows(sides, x, zero: float, top: float, bar: float, pitch: float, gap: float):
    """One panel's rows: the promoted tokens, then the suppressed ones, most suppressed last."""
    rows, y = [], top
    for side, pairs in sides:
        for i, (token, z) in enumerate(pairs):
            rank = i + 1 if side == "promoted" else len(pairs) - i
            shown = token.translate(VISIBLE_WHITESPACE)
            rows.append(
                {
                    "promoted": side == "promoted",
                    "y": y,
                    "d": _bar_path(zero, x(z), y, bar),
                    "text": shown if len(shown) <= 22 else shown[:21] + "…",
                    "text_x": zero - 6 if z >= 0 else zero + 6,
                    "anchor": "end" if z >= 0 else "start",
                    "label": f"{shown} · z = {z:+.2f} · #{rank} {side}",
                }
            )
            y += pitch
        y += gap
    return rows


def _vocab_table(lists: dict) -> list[dict]:
    """The same tokens as a table, one row per rank, for screen readers and copying."""
    return [
        {
            "rank": i + 1,
            **{
                f"{key}_{side}": lists[key, side][i] if i < len(lists[key, side]) else None
                for key, *_ in LENSES
                for side in ("promoted", "suppressed")
            },
        }
        for i in range(max(len(pairs) for pairs in lists.values()))
    ]


CAUSAL_KINDS = (("sufficiency", "Sufficiency"), ("necessity", "Necessity"))


def causal_chart(causal, counterfactuals, width: int = 640) -> dict | None:
    """Bar geometry: each pair's sufficiency and necessity percentile against the 0.5 line.

    A percentile ranks the unit's |effect| among the matched units' (protocol ≥ 0.5),
    so 0.5 is the baseline: as influential as a unit that changes as much. Each pair gets
    a row of two bars, and a final row holds the medians, which are the recorded scores.
    Records without per-pair effects get the median row alone. Records before 0.5 scored
    a fraction of the gap instead, which has no such baseline, so they get no chart.
    """
    if causal is None or causal.target is None:
        return None
    left, right = 132, width - 48
    bar, gap, row_h, top = 9, 2, 34, 30

    x = partial(_fraction_x, left=left, right=right)
    bars = partial(_causal_bars, x=x, left=left, bar=bar, gap=gap)
    rows = []
    for p in causal.pairs:
        y = top + len(rows) * row_h
        values = _pair_values(p, counterfactuals)
        rows.append({"label": f"Pair {p.pair_index + 1}", "y": y, "bars": bars(y, values)})
    median_y = top + len(rows) * row_h + (10 if rows else 0)
    detail = f" · median of {causal.pairs_scored} pairs" if causal.pairs_scored else ""
    medians = {
        "sufficiency": (causal.sufficiency_score, detail),
        "necessity": (causal.necessity_score, detail),
    }
    median = {"label": "Median (score)", "y": median_y, "bars": bars(median_y, medians)}
    axis_y = median_y + 2 * bar + gap + 10
    return {
        "width": width,
        "height": axis_y + 20,
        "left": left,
        "right": right,
        "top": top,
        "axis_y": axis_y,
        "rows": rows,
        "median": median,
        "rule_y": median_y - 8 if rows else None,
        "bar": bar,
        "baseline_x": x(0.5),
        "ticks": [(x(v), f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1)],
    }


def _fraction_x(v: float, *, left: float, right: float) -> float:
    return round(left + (right - left) * v, 2)


def _causal_bars(y: float, values: dict, *, x, left: float, bar: float, gap: float) -> list[dict]:
    """A row's two bars, sufficiency above necessity; ``values`` is kind -> (value, detail)."""
    out = []
    for i, (kind, label) in enumerate(CAUSAL_KINDS):
        value, detail = values[kind]
        out.append(
            {
                "kind": kind,
                "y": round(y + i * (bar + gap), 2),
                "w": round(x(value) - left, 2) if value is not None else None,
                "value": value,
                "label": f"{label}: " + ("—" if value is None else f"{value:.2f}") + detail,
            }
        )
    return out


def _pair_values(p, counterfactuals) -> dict[str, tuple[float | None, str]]:
    pair = counterfactuals[p.pair_index] if p.pair_index < len(counterfactuals) else None
    prompts = f"\n{pair.base_prompt} → {pair.counterfactual_prompt}" if pair else ""
    values = {}
    for kind, _ in CAUSAL_KINDS:
        patch = getattr(p, kind)
        values[kind] = (
            patch.percentile,
            f" · effect {patch.effect:.3g} (matched |effect| median "
            f"{patch.null_median:.3g}, max {patch.null_max:.3g}){prompts}",
        )
    return values


def _best_status(neuron) -> str:
    statuses = {st.status for st in neuron.standings.values()}
    return next(s for s in MAP_STATUSES if s in statuses)


def _poles(neuron) -> str:
    return ", ".join(f"{side} {st.status}" for side, st in neuron.standings.items()).replace(
        "_", " "
    )


def layer_map(model, width: int = 640) -> dict | None:
    """Layer × neuron-index geometry: one row per layer, one mark per studied neuron.

    Each mark sits at the neuron's index and takes its best pole's status (``MAP_STATUSES``
    order), so coverage and where the verified neurons sit both show. Marks are drawn worst
    status first, so a verified neuron is never hidden under a neighbour. Models whose MLP
    width no record gives get no map: the index axis would have no end.
    """
    if not model.mlp_width or not model.neurons:
        return None
    left, right = 36, width - 8
    row, pitch, top = 10, 13, 4
    mark_w = max(3.0, (right - left) / model.mlp_width)

    def x(idx: int) -> float:
        return round(left + (right - left - mark_w) * idx / max(model.mlp_width - 1, 1), 2)

    marks, counts = _layer_marks(model, x, top, pitch)
    axis_y = top + model.n_layers * pitch + 2
    step = _nice_step(model.mlp_width)
    ticks = [(x(i), f"{i:,}") for i in range(0, model.mlp_width, int(step))]
    return {
        "width": width,
        "height": axis_y + 18,
        "left": left,
        "right": right,
        "row": row,
        "mark_w": round(mark_w, 2),
        "axis_y": axis_y,
        "rows": [(i, top + i * pitch) for i in range(model.n_layers)],
        "label_every": 1 if model.n_layers <= 16 else 2,
        "marks": marks,
        "ticks": ticks,
        "counts": counts,
        "unexplored": model.total_neurons - model.neurons,
    }


def _layer_marks(model, x, top: float, pitch: float) -> tuple[list[dict], dict[str, int]]:
    """The map's marks, worst status first so the best draw on top, and a count per status."""
    counts = dict.fromkeys(MAP_STATUSES, 0)
    marks = []
    for layer in model.layers:
        y = top + layer.layer * pitch
        for n in layer.neurons:
            status = _best_status(n)
            counts[status] += 1
            marks.append(
                {
                    "x": x(n.idx),
                    "y": y,
                    "status": status,
                    "href": n.href,
                    "label": f"L{layer.layer} {n.label} · {_poles(n)}",
                }
            )
    marks.sort(key=lambda m: -MAP_STATUSES.index(m["status"]))
    return marks, counts


# Pieces bars: BitTorrent-style coverage, one cell per neuron or block of neurons. Cells sit
# on an integer pitch so the empty-cell grid can be one SVG pattern rather than a rect each.
PIECE_ROW = 128  # cells per bar
PIECE_PITCH = 5  # px per cell, 1px gap included


def model_pieces(model) -> dict | None:
    """One pieces bar per layer, each cell a block of ``ceil(mlp_width / PIECE_ROW)`` neurons.

    A cell is lit when any neuron in its block is studied, and takes the best status among
    them (``MAP_STATUSES`` order). With blocks this wide a lit cell is not a covered block:
    the bars show where the studied neurons are and where the gaps are, not how many.
    """
    if not model.mlp_width or not model.neurons:
        return None
    block = math.ceil(model.mlp_width / PIECE_ROW)
    n_cells = math.ceil(model.mlp_width / block)
    by_layer = {layer.layer: layer for layer in model.layers}
    rows = []
    for i in range(model.n_layers):
        cells: dict[int, list] = {}
        for n in by_layer[i].neurons if i in by_layer else ():
            cells.setdefault(n.idx // block, []).append(n)
        lit = []
        for c, neurons in sorted(cells.items()):
            status = min((_best_status(n) for n in neurons), key=MAP_STATUSES.index)
            lo, hi = c * block, min((c + 1) * block, model.mlp_width) - 1
            lit.append(
                {
                    "x": c * PIECE_PITCH,
                    "status": status,
                    "label": f"L{i} N{lo:,}–{hi:,}: "
                    + "; ".join(f"{n.label} {_poles(n)}" for n in neurons),
                }
            )
        rows.append({"layer": i, "cells": lit})
    return {
        "block": block,
        "pitch": PIECE_PITCH,
        "width": n_cells * PIECE_PITCH - 1,
        "rows": rows,
        "lit": sum(len(r["cells"]) for r in rows),
        "cells": n_cells * model.n_layers,
    }


def layer_pieces(model, layer) -> dict | None:
    """The layer's neurons as a grid of ``PIECE_ROW``-wide rows, one cell per neuron, so each
    studied neuron is a lit cell coloured by its best pole's status and linking to its page."""
    if not model.mlp_width:
        return None
    left, top = 8 * PIECE_PITCH, 0  # room for row labels; multiples of the pitch keep
    # the empty-cell pattern aligned with the lit cells
    n_rows = math.ceil(model.mlp_width / PIECE_ROW)
    cells = [
        {
            "x": left + (n.idx % PIECE_ROW) * PIECE_PITCH,
            "y": top + (n.idx // PIECE_ROW) * PIECE_PITCH,
            "status": _best_status(n),
            "href": n.href,
            "label": f"{n.label} · {_poles(n)}",
        }
        for n in layer.neurons
    ]
    rows = [
        {
            "y": top + r * PIECE_PITCH,
            "start": r * PIECE_ROW,
            "width": min(PIECE_ROW, model.mlp_width - r * PIECE_ROW) * PIECE_PITCH - 1,
        }
        for r in range(n_rows)
    ]
    counts = dict.fromkeys(MAP_STATUSES, 0)
    for c in cells:
        counts[c["status"]] += 1
    return {
        "left": left,
        "pitch": PIECE_PITCH,
        "width": left + PIECE_ROW * PIECE_PITCH,
        "height": top + n_rows * PIECE_PITCH,
        "rows": rows,
        "label_every": 8,
        "cells": cells,
        "counts": counts,
        "unexplored": model.mlp_width - len(cells),
    }
