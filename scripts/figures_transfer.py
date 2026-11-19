"""The figure of papers/explanations-across-models/README.md, from its results.jsonl."""

from figure_motion import reveal
from figure_svg import Figure, Panel, columns

# One colour and one shape per class through the paper: sisters, other matches, random poles.
# The shape keeps the classes apart without colour.
CLASSES = (("sister", "c1", "sisters"), ("match", "c3", "other top matches"),
           ("random", "c2", "random poles"))  # fmt: skip
SHAPES = {"sister": "circle", "match": "square", "random": "triangle"}
SLOT_WIDTH = 0.6  # of a class's column, over which its studies spread
CYCLE = "8s"  # the classes appear one after another, each median after its dots
# When each class's dots and then its median appear, in percent of the cycle.
DOTS_AT = {"sister": 8, "match": 30, "random": 52}
MEDIAN_AFTER = 12


def _mark(fig: Figure, name: str, x: float, y: float, size: float, cls: str) -> None:
    """One study's mark, in its class's shape (:data:`SHAPES`)."""
    shape = SHAPES[name]
    if shape == "circle":
        fig.dot(x, y, size, cls)
    elif shape == "square":
        fig.bar(x - size * 0.9, y - size * 0.9, size * 1.8, size * 1.8, cls)
    else:
        h = size * 1.15
        fig.path([(x, y - h), (x + h, y + h * 0.8), (x - h, y + h * 0.8)], cls, close=True)


def _shown(fig: Figure, name: str) -> tuple[str, str]:
    """The reveal classes of a class's dots and of its median, with their keyframes."""
    fig.motion += reveal(f"{name}dots", DOTS_AT[name], CYCLE)
    fig.motion += reveal(f"{name}mid", DOTS_AT[name] + MEDIAN_AFTER, CYCLE)
    return f"{name}dots", f"{name}mid"


def _strip(fig: Figure, panel: Panel, studies: list[dict]) -> None:
    """Each study's balanced accuracy in its class's column, with the class median."""
    panel.grid_y([0, 0.25, 0.5, 0.75, 1], 2)
    panel.horizontal(0.5, "axis")
    for slot, (name, colour, label) in enumerate(CLASSES):
        dots, median = _shown(fig, name)
        values = sorted(s["balanced_accuracy"] for s in studies if s["class"] == name)
        for i, value in enumerate(values):
            x = slot + 0.5 + SLOT_WIDTH * ((i % 7) / 6 - 0.5)  # a fixed spread, not random
            _mark(fig, name, panel.x(x), panel.y(value), 3.5, f"f{colour} {dots}")
        mid = values[len(values) // 2] if len(values) % 2 else sum(values[len(values) // 2 - 1:
              len(values) // 2 + 1]) / 2  # fmt: skip
        fig.line(panel.x(slot + 0.15), panel.y(mid), panel.x(slot + 0.85), panel.y(mid),
                 f"line s{colour} {median}")  # fmt: skip
        fig.text(panel.x(slot + 0.5), panel.top + panel.height + 15, label, "small", "middle")


def _scatter(fig: Figure, panel: Panel, studies: list[dict]) -> None:
    """Balanced accuracy against r, for the studies of sisters and other matches."""
    panel.grid_x([0, 0.25, 0.5, 0.75, 1], 2)
    panel.grid_y([0, 0.25, 0.5, 0.75, 1], 2, labels=False)
    panel.horizontal(0.5, "axis")
    panel.x_title("correlation r with the source pole")
    for name, colour, _ in CLASSES[:2]:
        for s in studies:
            if s["class"] == name and s["pearson"] is not None:
                _mark(fig, name, panel.x(s["pearson"]), panel.y(s["balanced_accuracy"]), 4,
                      f"f{colour} {name}dots")  # fmt: skip


def transfer(rows: list[dict]) -> Figure:
    """The two panels of the paper's figure."""
    studies = [r for r in rows if r["kind"] == "study" and r["balanced_accuracy"] is not None]
    fig = Figure(
        [
            "A pole's explanation carries better to its sister",
            "than to another match or a random pole",
        ],
        [
            "Each mark is a sister study: the balanced accuracy of the source's explanation "
            "on the tested pole.",
            "Circles are sisters, squares other matches and triangles random poles. "
            "The line marks chance.",
        ],
        "Two plots. Left: balanced accuracy by class, with medians of 0.78 for sisters, 0.55 "
        "for other matches and 0.45 for random poles. Right: balanced accuracy against the "
        "correlation r for sisters and other matches.",
    )
    left, right = columns(fig, 2, 60, fig.body_top + 34, 220, (0, 3), (0, 1), gap=36)
    left.x_range = (0, 3)
    left.head("by class")
    right.x_range = (0, 1)
    right.head("against r")
    _strip(fig, left, studies)
    _scatter(fig, right, studies)
    bottom = left.top + left.height + 48
    x = 20.0
    for name, colour, label in CLASSES:  # the legend in each class's shape, not only colour
        _mark(fig, name, x + 5, bottom - 4, 5, f"f{colour}")
        fig.text(x + 15, bottom, label)
        x += 26 + 6.6 * len(label)
    fig.bottom = bottom
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name."""
    return {"transfer.svg": transfer(rows)}
