"""The figure of papers/neuron-development, drawn from its results (issue #3376)."""

import math

from figure_motion import animate, counter, even_stops, translate
from figure_svg import Figure, Panel

TICKS = (1, 10, 100, 1000, 10_000, 100_000)
CYCLE = "9s"
HOLD_FROM = 80  # percent of the cycle after which every moving dot rests on the last step


def _x(step: int) -> float:
    """A step on a log axis. Step 0 sits at 1, the left end."""
    return math.log10(step + 1)


def _label(step: int) -> str:
    return f"{step // 1000}k" if step >= 1000 else str(step)


def _panels(fig: Figure) -> tuple[Panel, Panel]:
    """The correlation panel and the loss panel, with their grids, labels and the 16k line."""
    span = (0.0, _x(150_000))
    left = Panel(fig, (64, fig.body_top + 28, 400, 260), span, (-0.2, 1.0))
    right = Panel(fig, (520, fig.body_top + 28, 216, 260), span, (3.0, 11.5))
    for panel, head in ((left, "Correlation with the final form"), (right, "Loss (nats)")):
        panel.head(head)
        for tick in TICKS:
            panel.figure.line(panel.x(_x(tick)), panel.top, panel.x(_x(tick)),
                              panel.top + panel.height, "grid")  # fmt: skip
            panel.figure.text(panel.x(_x(tick)), panel.top + panel.height + 15, _label(tick),
                              "small", "middle")  # fmt: skip
        panel.x_title("training step (log scale)")
    left.grid_y((0.0, 0.5, 1.0), places=1)
    right.grid_y((4, 6, 8, 10), places=0)
    left.vertical(_x(16_000), "dip")
    right.vertical(_x(16_000), "dip")
    fig.text(right.x(_x(16_000)) - 6, right.y(10.8), "step 16k: loss", "small", "end")
    fig.text(right.x(_x(16_000)) - 6, right.y(10.8) + 14, "within 1% of final", "small", "end")
    return left, right


DASH = ".dash{stroke-dasharray:6 4}"


def _positive(poles: list[dict]) -> list[dict]:
    """The positive poles' rows. A negative pole at GELU's floor has no correlation (paper)."""
    return [r for r in poles if r["slug"].endswith("-positive")]


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """The lower quartile, median and upper quartile, by the nearest rank."""
    xs = sorted(values)
    n = len(xs)
    median = (xs[(n - 1) // 2] + xs[n // 2]) / 2
    return xs[n // 4], median, xs[(3 * n) // 4]


def _bands(rows: list[dict], key: str) -> list[tuple[int, float, float, float]]:
    """``(step, lower quartile, median, upper quartile)`` of ``key`` over the poles, by step."""
    steps = sorted({r["step"] for r in rows if r[key] is not None})
    return [(st, *_quartiles([r[key] for r in rows if r["step"] == st and r[key] is not None]))
            for st in steps]  # fmt: skip


def _band(panel: Panel, bands: list[tuple]) -> None:
    """The quartiles as a shaded band, and the median as a line."""
    upper = [(panel.x(_x(st)), panel.y(hi)) for st, _, _, hi in bands]
    lower = [(panel.x(_x(st)), panel.y(lo)) for st, lo, _, _ in reversed(bands)]
    panel.figure.shaded(upper + lower, "fwash", 0.9)
    panel.curve([_x(b[0]) for b in bands], [b[2] for b in bands], "curve")


def holds_from(rows: list[dict]) -> dict[str, int | None]:
    """Each pole's first step from which its probes pass the gates at every later step."""
    out: dict[str, int | None] = {}
    for slug in sorted({r["slug"] for r in rows}):
        own = sorted((r for r in rows if r["slug"] == slug), key=lambda r: r["step"])
        out[slug] = next((r["step"] for i, r in enumerate(own)
                          if all(x["holds"] for x in own[i:])), None)  # fmt: skip
    return out


def first_poles(rows: list[dict]) -> set[str]:
    """The slugs of the first run's poles: the ``first`` row's list."""
    return set(next(r for r in rows if r["table"] == "first")["slugs"])


def development(rows: list[dict]) -> Figure:
    """The poles' median correlation with their final form by step, next to the loss."""
    poles = _positive([r for r in rows if r["table"] == "pole"])
    loss = [r for r in rows if r["table"] == "loss"]
    first = [r for r in poles if r["slug"] in first_poles(rows)]
    n = len({r["slug"] for r in poles})
    fig = Figure(
        ["Training loss settles by step 16,000, but the VERIFIED poles keep changing"],
        [
            f"Left: the median correlation of {n} Pythia-70m poles with their final activations,",
            "by step. The band holds the middle half of the poles. Dashed: the first 9 poles.",
            "Right: next-token loss on the same 300 windows.",
            "In each panel, a moving dot steps through the checkpoints.",
        ],
        f"A band of correlations with the final checkpoint for {n} poles, whose median rises "
        "slowly to 1 by step 143,000, beside a loss curve that is flat from about step 16,000.",
    )
    fig.motion += DASH
    left, right = _panels(fig)
    bands = _bands(poles, "r_final")
    _band(left, bands)
    old = _bands(first, "r_final")
    left.curve([_x(b[0]) for b in old], [b[2] for b in old], "line sneutral dash")
    fig.text(left.x(_x(4_000)), left.y(0.52), f"{n} poles, median", "small", "end")
    fig.text(left.x(_x(64_000)) + 6, left.y(0.64), "first 9", "small")
    loss = sorted(loss, key=lambda r: r["step"])
    right.curve([_x(r["step"]) for r in loss], [r["loss"] for r in loss], "curve")
    medians = [(b[0], b[2]) for b in bands]
    _animate(fig, left, right, medians, loss)
    fig.bottom = left.top + left.height + 36
    return fig


def holding(rows: list[dict]) -> Figure:
    """When each explanation starts to hold, and how far each pole moves between checkpoints."""
    every = [r for r in rows if r["table"] == "pole"]
    start = holds_from(every)
    born = [{"slug": slug, "step": step} for slug, step in start.items()]
    first = first_poles(rows)
    poles = _positive(every)
    fig = Figure(
        ["Half the explanations hold only after the loss settles, and the poles keep moving"],
        [
            f"Left: the share of the {len(born)} explanations that hold without a break from each",
            "step on. Dashed: the first 9. Right: each pole's correlation with the checkpoint",
            "before; the median, and a band for the middle half of the poles.",
        ],
        "A step curve of the share of explanations that hold from each step, which passes "
        "half at step 32,000, beside a band of adjacent-checkpoint correlations near 0.8 after "
        "step 16,000.",
    )
    fig.motion += DASH
    left, right = _holding_panels(fig)
    steps = sorted({r["step"] for r in rows if r["table"] == "loss"})
    _held(left, born, steps, "curve")
    _held(left, [r for r in born if r["slug"] in first], steps, "line sneutral dash")
    _band(right, _bands(poles, "r_prev"))
    fig.text(left.x(_x(3_000)), left.y(0.3), f"all {len(born)}", "small", "end")
    fig.text(left.x(_x(30_000)), left.y(0.16), "first 9", "small")
    fig.bottom = left.top + left.height + 36
    return fig


def _holding_panels(fig: Figure) -> tuple[Panel, Panel]:
    """Figure 2's two panels, each a share from 0 to 1 over the log steps, with the 16k line."""
    span = (0.0, _x(150_000))
    left = Panel(fig, (64, fig.body_top + 28, 300, 240), span, (0.0, 1.0))
    right = Panel(fig, (436, fig.body_top + 28, 300, 240), span, (0.0, 1.0))
    for panel, head in ((left, "Share whose explanation holds from here on"),
                        (right, "Correlation with the checkpoint before")):  # fmt: skip
        panel.head(head)
        for tick in TICKS:
            fig.line(panel.x(_x(tick)), panel.top, panel.x(_x(tick)), panel.top + panel.height,
                     "grid")  # fmt: skip
            fig.text(panel.x(_x(tick)), panel.top + panel.height + 15, _label(tick), "small",
                     "middle")  # fmt: skip
        panel.x_title("training step (log scale)")
        panel.grid_y((0.0, 0.5, 1.0), places=1)
        panel.vertical(_x(16_000), "dip")
    return left, right


def _held(panel: Panel, group: list[dict], steps: list[int], cls: str) -> None:
    """The share of ``group`` that holds from each step on, as a step curve."""
    share = [sum(r["step"] is not None and r["step"] <= st for r in group) / len(group)
             for st in steps]  # fmt: skip
    xs, ys = [], []
    for i, st in enumerate(steps):  # flat until the next checkpoint
        if i:
            xs.append(_x(st))
            ys.append(share[i - 1])
        xs.append(_x(st))
        ys.append(share[i])
    panel.curve(xs, ys, cls)


def _animate(fig: Figure, left: Panel, right: Panel, medians: list, loss: list[dict]) -> None:
    """Dots that walk the median correlation and the loss through the measured steps.

    Each frame is a measured checkpoint, spread evenly over the cycle, with a counter that
    names the step. The dots jump between checkpoints, so they show no value that was not
    measured. With motion off, every dot rests on the last step.
    """
    steps = sorted({r["step"] for r in loss})
    tracks = [(right, "lossdot", [(r["step"], r["loss"]) for r in loss]),
              (left, "mediandot", medians)]  # fmt: skip
    for panel, name, points in tracks:
        stops = even_stops([translate(panel.x(_x(st)), panel.y(v)) for st, v in points], HOLD_FROM)
        # step-end: each dot jumps from checkpoint to checkpoint, never between them.
        last = stops[-1][1]
        fig.motion += animate(name, stops, CYCLE, timing="step-end", hold=last, rest=last)
        fig.dot(0, 0, 4, f"ink ring {name}")
    starts = [float(at[:-1]) for at, _ in even_stops([""] * len(steps), HOLD_FROM)]
    for (cls, css), step in zip(counter("stepnow", starts, CYCLE), steps, strict=True):
        fig.motion += css
        fig.text(left.x(_x(steps[-1])), left.top - 8, f"step {step:,}", f"small {cls}", "end")


def build(rows: list[dict]) -> dict[str, Figure]:
    """The paper's figures by file name."""
    from figures_development_savings import figure as savings_figure

    return {
        "development.svg": development(rows),
        "holding.svg": holding(rows),
        "savings.svg": savings_figure(rows),
    }
