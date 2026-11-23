"""The figures of papers/sequential-tuning/README.md, from its results.jsonl (#5125)."""

import math
from itertools import pairwise

from figure_motion import counter
from figure_svg import Figure, Panel, columns, number

# One colour and one shape per model through the paper. The shape keeps them apart without
# colour: circles are the base model, triangles the instruct model.
MODELS = (("base", "c1", "Qwen2.5-0.5B (base)"), ("instruct", "c2", "Qwen2.5-0.5B-Instruct"))
STAGES = ("start", "after SST-2", "after XSum", "after GSM8K")
TASKS = (("sst2", "SST-2 sentiment"), ("xsum", "XSum summary"), ("gsm8k", "GSM8K maths"))
CYCLE = "12s"
# Where each stage's label sits beside its mark in the KL figure, so that close marks keep
# apart: (dx, dy, anchor) in pixels. A stage not listed sits right of and above its mark.
PLACES = {
    ("instruct", "sst2"): (9, 14, "start"),
    ("instruct", "xsum"): (9, 14, "start"),
    ("base", "gsm8k"): (9, 14, "start"),
    ("instruct", "gsm8k"): (0, -12, "middle"),
}
HALF = ".half{opacity:.6}"  # dense marks, see-through so that overlaps show


def _mark(fig: Figure, model: str, x: float, y: float, size: float, style: str) -> None:
    """One mark in the model's shape: a circle for the base model, a triangle for instruct.

    Args:
        fig: The figure.
        model: ``base`` or ``instruct``.
        x: The mark's centre, in pixels.
        y: The mark's centre, in pixels.
        size: The mark's radius, in pixels.
        style: The mark's CSS classes.
    """
    if model == "base":
        fig.dot(x, y, size, style)
    else:
        h = size * 1.2
        fig.path([(x, y - h), (x + h, y + h * 0.8), (x - h, y + h * 0.8)], style, close=True)


def _legend(fig: Figure, y: float) -> None:
    """Each model's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for model, colour, label in MODELS:
        _mark(fig, model, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, label)
        x += 30 + 6.4 * len(label)


def _stages(rows: list[dict], arm: str) -> list[dict]:
    """The ``stage`` rows of one arm, in stage order.

    Args:
        rows: The paper's results.
        arm: ``base`` or ``instruct``.

    Returns:
        The rows.
    """
    return sorted((r for r in rows if r["table"] == "stage" and r["arm"] == arm),
                  key=lambda r: r["stage"])  # fmt: skip


def _bounds(values: list[float], step: float) -> tuple[float, float]:
    """A range that holds ``values``, widened to whole multiples of ``step``, never empty.

    Args:
        values: The values.
        step: The tick step.

    Returns:
        ``(low, high)``.

    Examples:
        >>> _bounds([0.62, 0.71], 0.05)
        (0.6, 0.75)
        >>> _bounds([0, 66], 10.0)
        (0.0, 70.0)
    """
    low = math.floor(min(values) / step + 1e-9) * step
    high = max(math.ceil(max(values) / step - 1e-9) * step, low + step)
    return round(low, 6), round(high, 6)


def _ticks(low: float, high: float, step: float) -> list[float]:
    """Each multiple of ``step`` from ``low`` to ``high``.

    Args:
        low: The first tick.
        high: The last tick.
        step: The step.

    Returns:
        The ticks.

    Examples:
        >>> _ticks(0.6, 0.75, 0.05)
        [0.6, 0.65, 0.7, 0.75]
    """
    return [round(low + i * step, 6) for i in range(round((high - low) / step) + 1)]


def _by_stage(panel: Panel, rows: list[dict], value: str, interval: str | None,
              places: int) -> None:  # fmt: skip
    """One line per model across the stages, with a 95% interval at each stage.

    Args:
        panel: The panel. Its x axis is the stage.
        rows: The paper's results.
        value: The stage rows' field to draw.
        interval: The field of its interval, or None for no interval.
        places: The decimal places of the end labels.
    """
    fig = panel.figure
    for k, (model, colour, _) in enumerate(MODELS):
        got = _stages(rows, model)
        shift = (k - 0.5) * 0.08  # the two models' marks side by side, not on top
        xs = [r["stage"] + shift for r in got]
        ys = [r[value] for r in got]
        panel.curve(xs, ys, f"line s{colour}")
        for r, x, y in zip(got, xs, ys, strict=True):
            if interval:
                lo, hi = r[interval]
                fig.line(panel.x(x), panel.y(lo), panel.x(x), panel.y(hi), f"line s{colour}")
            _mark(fig, model, panel.x(x), panel.y(y), 4.5, f"f{colour} ring")
        lift = -4 if model == "base" else 12  # base above, instruct below: no overlap
        fig.text(panel.x(xs[-1]) + 10, panel.y(ys[-1]) + lift, number(ys[-1], places), "ink")
    for stage, label in enumerate(STAGES):
        fig.text(panel.x(stage), panel.top + panel.height + 15, label, "small", "middle")


def stages(rows: list[dict]) -> Figure:
    """The keep share and the ARC-Easy accuracy after each stage: the paper's head figure.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    head = next(r for r in rows if r["table"] == "summary")
    ends = {m: _stages(rows, m)[-1] for m, _, _ in MODELS}
    fig = Figure(
        ["After three narrow fine-tunes, a quarter of the",
         "explanations fail, and ARC-Easy ends near its start"],
        [f"{head['reproduced']} VERIFIED poles of Qwen2.5-0.5B, each re-read on its own probes "
         "after each stage.", "Left: the share that still passes both gates. Right: zero-shot "
         "ARC-Easy accuracy. Bars: 95% intervals."],
        "Two line charts over four stages, one line per model. The share of explanations that "
        f"keep ends at {number(ends['base']['keep_share'], 3)} for the base model and "
        f"{number(ends['instruct']['keep_share'], 3)} for the instruct model. ARC-Easy accuracy "
        f"ends at {number(ends['base']['arc_accuracy'], 3)} and "
        f"{number(ends['instruct']['arc_accuracy'], 3)}.",
    )  # fmt: skip
    top = fig.body_top + 30
    left, right = columns(fig, 2, 60, top, 230, (-0.4, 3.6), (0, 1), gap=70, right=50)
    for panel, value, interval, step, label in (
            (left, "keep_share", "keep_ci95", 0.1, "share of explanations that keep"),
            (right, "arc_accuracy", "arc_ci95", 0.05, "ARC-Easy accuracy")):  # fmt: skip
        values = [v for r in rows if r["table"] == "stage" for v in r[interval]]
        panel.y_range = _bounds(values, step)
        panel.grid_y(_ticks(*panel.y_range, step), 2)
        panel.head(label)
        _by_stage(panel, rows, value, interval, 3)
    bottom = top + 230 + 50
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def losses(rows: list[dict]) -> Figure:
    """Each task's held-out loss after each stage, one panel per task.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    fig = Figure(
        ["Each stage learns its task, and the later",
         "stages undo part of the earlier ones"],
        ["The mean loss, in nats, on the response tokens of 200 held-out pairs of each task, "
         "after", "each stage. A stage trains on one task, in the order SST-2, XSum, GSM8K."],
        "Three line charts, one per task, each with one line per model over four stages. "
        "Each task's loss falls at its own stage. The SST-2 loss rises again at the later "
        "stages.",
    )  # fmt: skip
    top = fig.body_top + 30
    values = [v for r in rows if r["table"] == "stage" for v in r["task_loss"].values()]
    low, high = _bounds(values, 1.0)
    panels = columns(fig, 3, 50, top, 200, (-0.4, 3.4), (low, high), gap=40, right=40)
    for panel, (task, label) in zip(panels, TASKS, strict=True):
        panel.grid_y(_ticks(low, high, 1.0), 0, labels=panel is panels[0])
        panel.head(label)
        fig = panel.figure
        for model, colour, _ in MODELS:
            got = _stages(rows, model)
            ys = [r["task_loss"][task] for r in got]
            panel.curve(range(len(ys)), ys, f"line s{colour}")
            for x, y in enumerate(ys):
                _mark(fig, model, panel.x(x), panel.y(y), 4, f"f{colour} ring")
        for stage in range(len(STAGES)):
            fig.text(panel.x(stage), top + 215, str(stage), "small", "middle")
        panel.x_title("stages done")
    bottom = top + 200 + 60
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def kl(rows: list[dict]) -> Figure:
    """Each stage's KL on its own data against the explanations that it costs.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    points = []
    for model, _, _ in MODELS:
        got = _stages(rows, model)
        for before, after in pairwise(got):
            points.append((model, after["task"], after["kl"],
                           before["keeps"] - after["keeps"]))  # fmt: skip
    fig = Figure(
        ["The stage that moves its own data most is",
         "not the stage that costs the most explanations"],
        ["Each mark is one stage of one model: KL(before ‖ after) on the stage's held-out "
         "responses,", "in nats (across), against the net number of explanations that stop "
         "keeping at that stage (up)."],
        f"A scatter plot of {len(points)} marks, labelled by task: "
        + "; ".join(f"{m} {t}: KL {number(k, 2)}, {d} lost" for m, t, k, d in points) + ".",
    )  # fmt: skip
    xs = [p[2] for p in points]
    ys = [p[3] for p in points]
    x_low, x_high = _bounds([0, *xs], 2.0)
    y_low, y_high = _bounds([0, *ys], 10.0)
    panel = Panel(fig, (90, fig.body_top + 24, 520, 260), (x_low, x_high), (y_low, y_high))
    panel.grid_x(_ticks(x_low, x_high, 2.0), 0)
    panel.grid_y(_ticks(y_low, y_high, 10.0), 0)
    panel.horizontal(0, "axis")
    panel.x_title("KL on the stage's own held-out data (nats)")
    fig.text(panel.left - 50, panel.top - 8, "explanations lost at the stage", "small")
    names = dict(TASKS)
    for model, task, x, y in points:
        colour = dict((m, c) for m, c, _ in MODELS)[model]
        _mark(fig, model, panel.x(x), panel.y(y), 5, f"f{colour} ring")
        dx, dy, anchor = PLACES.get((model, task), (9, -6, "start"))
        fig.text(panel.x(x) + dx, panel.y(y) + dy, names[task].split()[0], "small", anchor)
    bottom = panel.top + panel.height + 56
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def _places(slugs: list[tuple[int, str]]) -> tuple[list[tuple[int, int]], int]:
    """Each pole's column and row in the grid, and the deepest column's length.

    Args:
        slugs: ``(layer, slug)`` of each pole, in grid order.

    Returns:
        ``([(layer, row), ...], deepest)``.

    Examples:
        >>> _places([(0, "a"), (0, "b"), (2, "c")])
        ([(0, 0), (0, 1), (2, 0)], 2)
    """
    seen: dict[int, int] = {}
    places = []
    for layer, _ in slugs:
        places.append((layer, seen.get(layer, 0)))
        seen[layer] = places[-1][1] + 1
    return places, max(seen.values())


def grid(rows: list[dict]) -> Figure:
    """Every pole, by layer, at each stage of each model in turn.

    Args:
        rows: The paper's results.

    Returns:
        The figure. It steps through the base model's stages, then the instruct model's,
        and rests on the instruct model after the last stage.
    """
    poles = [r for r in rows if r["table"] == "pole"]
    kept = {r["slug"] for r in poles if (r["arm"], r["stage"]) == ("base", 0) and r["keeps"]}
    reading = {(r["arm"], r["stage"], r["slug"]): r["keeps"] for r in poles}
    slugs = sorted({(r["layer"], r["slug"]) for r in poles if r["slug"] in kept})
    layers = max(layer for layer, _ in slugs) + 1
    fig = Figure(
        ["Explanations fail in every layer, most in the",
         "middle, and many come back at a later stage"],
        ["Each mark is one VERIFIED pole of Qwen2.5-0.5B, in the column of its layer. A filled "
         "mark", "passes its own probes, and an open mark fails. The frames step through each "
         "model's stages."],
        f"A grid of {len(slugs)} marks in {layers} columns, one per layer. Eight frames: the "
        "base model at the start and after each of three fine-tunes, then the instruct model "
        "the same way. Open marks appear in every third of the layers, most of all in layers "
        "8 to 15, and some marks that open at one stage fill again at a later one.",
    )  # fmt: skip
    fig.motion += HALF
    left, top, step = 70.0, fig.body_top + 40, 8.6
    places, deepest = _places(slugs)
    spot = lambda i: (left + places[i][0] * 27.5, top + places[i][1] * step)  # noqa: E731
    frames = [(m, c, s) for m, c, _ in MODELS for s in range(len(STAGES))]
    starts = [100 * i / len(frames) * 0.85 for i in range(len(frames))]
    for (cls, css), (model, colour, stage) in zip(counter("stage", starts, CYCLE), frames,
                                                  strict=True):  # fmt: skip
        fig.motion += css
        keeps = sum(reading[(model, stage, s)] for _, s in slugs)
        name = "base model" if model == "base" else "instruct model"
        fig.text(left - 6, top - 18, f"{name}, {STAGES[stage]}: {keeps} of {len(slugs)} keep",
                 f"head {cls}")  # fmt: skip
        for i, (_, slug) in enumerate(slugs):
            x, y = spot(i)
            style = f"f{colour}" if reading[(model, stage, slug)] else f"open s{colour}"
            _mark(fig, model, x, y, 3.2, f"{style} {cls}")
    for layer in range(0, layers, 4):
        fig.text(left + layer * 27.5, top + deepest * step + 12, f"L{layer}", "small", "middle")
    bottom = top + deepest * step + 44
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"stages.svg": stages(rows), "grid.svg": grid(rows), "losses.svg": losses(rows),
            "kl.svg": kl(rows)}  # fmt: skip
