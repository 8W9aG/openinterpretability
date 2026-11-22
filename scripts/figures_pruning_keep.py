"""The figures of papers/pruning-keep/README.md, from its results.jsonl (#5122)."""

from figure_motion import counter
from figure_svg import Figure, Panel, number

# One colour and one shape per method through the paper: circles are magnitude pruning,
# triangles the random null. The shape keeps them apart without colour.
METHODS = (("magnitude", "c1", "magnitude (smallest weights first)"),
           ("random", "c2", "random (the null)"))  # fmt: skip
CYCLE = "8s"
HALF = ".half{opacity:.6}"  # dense dots, see-through so that overlaps show


def _mark(fig: Figure, method: str, x: float, y: float, size: float, style: str) -> None:
    """One mark in the method's shape: a circle for magnitude, a triangle for random.

    Args:
        fig: The figure.
        method: ``magnitude`` or ``random``.
        x: The mark's centre, in pixels.
        y: The mark's centre, in pixels.
        size: The mark's radius, in pixels.
        style: The mark's CSS classes.
    """
    if method == "magnitude":
        fig.dot(x, y, size, style)
    else:
        h = size * 1.2
        fig.path([(x, y - h), (x + h, y + h * 0.8), (x - h, y + h * 0.8)], style, close=True)


def _legend(fig: Figure, y: float) -> None:
    """Each method's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for method, colour, label in METHODS:
        _mark(fig, method, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, label)
        x += 30 + 6.4 * len(label)


def _summary(rows: list[dict]) -> dict:
    """The ``summary`` row.

    Args:
        rows: The paper's results.

    Returns:
        The row.
    """
    return next(r for r in rows if r["table"] == "summary")


def keep(rows: list[dict]) -> Figure:
    """The share of explanations that keep, by pruning level, for each method.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    head = _summary(rows)
    levels = [r for r in rows if r["table"] == "level"]
    last = {r["method"]: r for r in levels if r["fraction"] == max(head["levels"])}
    fig = Figure(
        ["Magnitude pruning keeps most explanations, and",
         "random pruning of as many weights does not"],
        [f"Each mark is the share of Pythia-70m's {head['reproduced']} VERIFIED explanations that "
         "still pass their probes, with a 95% interval.",
         f"With 90% of each neuron's input weights zeroed: {last['magnitude']['keeps']} keep "
         f"by magnitude, {last['random']['keeps']} at random."],
        f"A line chart of the keep share against the share of input weights zeroed: 0, 50, 75 "
        f"and 90%. Magnitude pruning keeps {number(last['magnitude']['keep_share'], 2)} at 90%, "
        f"and random pruning {number(last['random']['keep_share'], 2)}. Both start at 1.",
    )  # fmt: skip
    panel = Panel(fig, (70, fig.body_top + 24, 640, 230), (-0.03, 0.93), (0, 1.05))
    panel.grid_x([0, 0.25, 0.5, 0.75, 0.9], 2)
    panel.grid_y([0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.x_title("share of each neuron's input weights zeroed")
    for method, colour, _ in METHODS:
        points = [(0.0, 1.0, 1.0, 1.0)] + [
            (r["fraction"], r["keep_share"], *r["keep_ci95"])
            for r in sorted(levels, key=lambda r: r["fraction"]) if r["method"] == method
        ]  # fmt: skip
        panel.curve([p[0] for p in points], [p[1] for p in points], f"line s{colour}")
        for x, share, low, high in points:
            fig.line(panel.x(x), panel.y(low), panel.x(x), panel.y(high), f"range s{colour}")
            _mark(fig, method, panel.x(x), panel.y(share), 5, f"f{colour} ring")
    bottom = panel.top + panel.height + 50
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def pearson(rows: list[dict]) -> Figure:
    """Each pole's correlation with its unpruned self, one frame per pruning level.

    Args:
        rows: The paper's results.

    Returns:
        The figure. It animates through the levels and rests on the last.
    """
    head = _summary(rows)
    poles = [r for r in rows if r["table"] == "pole" and r["method"] != "none"]
    fig = Figure(
        ["As more weights go, magnitude pruning keeps each",
         "pole's activations, and random pruning does not"],
        ["Each mark is one pole: Pearson's r between its activations before and after pruning,",
         "on 20,000 tokens of web text. The frames step through the pruning levels."],
        "Two strips of 127 marks each, for magnitude and random pruning, at 50, 75 and 90% of "
        "the input weights zeroed. The magnitude marks stay near r = 1 longer. The random "
        "marks spread down towards 0.",
    )  # fmt: skip
    fig.motion += HALF
    panel = Panel(fig, (150, fig.body_top + 30, 560, 140), (-0.2, 1.0), (0, 2))
    panel.grid_x([0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.x_title("Pearson's r with the unpruned activations")
    starts = [100 * i / len(head["levels"]) for i in range(len(head["levels"]))]
    for (cls, css), fraction in zip(counter("level", starts, CYCLE), head["levels"], strict=True):
        fig.motion += css
        fig.text(panel.left + panel.width, panel.top - 8,
                 f"{round(fraction * 100)}% zeroed", f"head {cls}", "end")  # fmt: skip
        for slot, (method, colour, _) in enumerate(METHODS):
            centre = 1.5 - slot
            if fraction == head["levels"][0]:
                fig.text(panel.left - 10, panel.y(centre) + 4, method, "small", "end")
            got = sorted(r["pearson"] for r in poles
                         if r["fraction"] == fraction and r["method"] == method
                         and r["pearson"] is not None)  # fmt: skip
            for i, r in enumerate(got):
                jitter = ((i * 37) % 11 / 10 - 0.5) * 0.6  # a fixed spread, not random
                _mark(fig, method, panel.x(max(r, -0.2)), panel.y(centre + jitter), 2.6,
                      f"f{colour} half {cls}")  # fmt: skip
    bottom = panel.top + panel.height + 50
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
    return {"keep.svg": keep(rows), "pearson.svg": pearson(rows)}
