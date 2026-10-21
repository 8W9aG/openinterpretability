"""Static SVG chart geometry for experiment pages, computed at build time.

Each function returns plain coordinates and labels; the templates draw them.
"""

from __future__ import annotations

import math

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
    tests = [
        t
        for t in verdict.adversarial_test_results
        if t.kind in {kind for kind, *_ in PROBE_ROWS} and t.activation is not None
    ]
    if not tests:
        return None
    threshold = verdict.policy.activation_threshold * scale
    lo = min(0.0, *(t.activation for t in tests))
    hi = max(scale, *(t.activation for t in tests))
    left, right = 132, width - 12
    r, row_gap, top = 5, 64, 34

    def x(v: float) -> float:
        return round(left + (right - left) * (v - lo) / ((hi - lo) or 1.0), 2)

    rows = []
    for i, (kind, label, expect) in enumerate(PROBE_ROWS):
        cy = top + 24 + i * row_gap
        placed: list[tuple[float, float]] = []
        dots = []
        group = sorted((t for t in tests if t.kind == kind), key=lambda t: t.activation)
        for t in group:
            cx = x(t.activation)
            for dy in (0, -10, 10, -20, 20):
                if all(
                    (cx - px) ** 2 + (cy + dy - py) ** 2 >= (2 * r + 1) ** 2 for px, py in placed
                ):
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
        passed = sum(d["passed"] for d in dots)
        rows.append(
            {
                "label": label,
                "expect": expect,
                "y": cy,
                "dots": dots,
                "passed": passed,
                "n": len(dots),
            }
        )
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

    def x(v: float) -> float:
        return round(left + (right - left) * v, 2)

    def bars(y: float, values: dict[str, tuple[float | None, str]]) -> list[dict]:
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

    rows = []
    for p in causal.pairs:
        pair = counterfactuals[p.pair_index] if p.pair_index < len(counterfactuals) else None
        prompts = f"\n{pair.base_prompt} → {pair.counterfactual_prompt}" if pair else ""
        y = top + len(rows) * row_h
        values = {}
        for kind, _ in CAUSAL_KINDS:
            patch = getattr(p, kind)
            values[kind] = (
                patch.percentile,
                f" · effect {patch.effect:.3g} (matched |effect| median "
                f"{patch.null_median:.3g}, max {patch.null_max:.3g}){prompts}",
            )
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
