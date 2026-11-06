"""The figures of the analyses that use depth: ridges of h by layer, and a surface of the rest."""

from collections.abc import Callable
from statistics import mean

from figure_svg import Figure, normal, number, sample
from figures_default_on import _rest_rows
from figures_why_rest import LONG_STEPS, _rest, _run

Point = tuple[float, float]
SPAN = (-3.6, 2.2)
SCALE = 560 / (SPAN[1] - SPAN[0])
RIDGE_CYCLE = "5s"
LAYER_RISE = 15.0
LAYER_SHIFT = 7.0
PEAK = 70.0


def _ridge(fig: Figure, layer: int, start: Point, end: Point, geometry: Point) -> None:
    """Draw one layer's curve, moving between its random-init and its trained place."""
    left, base = geometry
    back = 11 - layer
    floor = base - back * LAYER_RISE
    shift = back * LAYER_SHIFT

    def frame(place: Point) -> str:
        x = left + (place[0] - SPAN[0]) * SCALE + shift
        return (
            f"transform:translate({x:.1f}px,{floor:.1f}px) scale({place[1]:.3f},{1 / place[1]:.3f})"
        )

    x_low, x_high = left + shift, left + (SPAN[1] - SPAN[0]) * SCALE + shift
    fig.line(x_low, floor, x_high, floor, "axis")
    units = sample(-3.4, 3.4, count=61)
    curve = [(u * SCALE, -PEAK * normal(u, 0, 1)) for u in units]
    name = f"ridge{layer}"
    fig.motion += (
        f"@keyframes {name}{{0%,25%{{{frame(start)}}}75%,100%{{{frame(end)}}}}}"
        f".{name}{{animation:{name} {RIDGE_CYCLE} ease-in-out infinite alternate;{frame(end)}}}"
    )
    fig.marks.append(f'<g class="ridge {name}">')
    fig.path(curve, "bg2", close=True)
    fig.path(curve, "fc1 soft", close=True)
    fig.path(curve, "line sc1")
    fig.marks.append("</g>")
    if layer in (0, 1, 11):
        fig.text(x_low - 6, floor + 4, f"L{layer}", "small", "end")


def _phase_labels(fig: Figure, x: float, y: float) -> None:
    """Add the two names of the moving curves' states. Each shows only in its own state."""
    fig.text(x, y, "Random init (mean of 5 seeds)", "head phase-a")
    fig.text(x, y, "Trained", "head phase-b")
    fig.motion += (
        "@keyframes phase-a{0%,40%{opacity:1}60%,100%{opacity:0}}"
        "@keyframes phase-b{0%,40%{opacity:0}60%,100%{opacity:1}}"
        f".phase-a{{opacity:0;animation:phase-a {RIDGE_CYCLE} linear infinite alternate}}"
        f".phase-b{{animation:phase-b {RIDGE_CYCLE} linear infinite alternate}}"
    )


def ridges(rows: list[dict], dip: float) -> Figure:
    """Draw h by layer in depth, moving between random init and the trained model."""
    trained, fresh = _rest_rows(rows)
    fig = Figure(
        ["Every layer but L1 slides past the dip together"],
        [
            "A sketch in depth: one normal curve for each layer, with its median h and its median"
            " neuron's SD.",
            "Only the two end states are measured. The dark line on the floor is GELU's dip, at"
            f" {number(dip)}.",
        ],
        "Twelve ridges, one for each layer of GPT-2 small, moving between random init, where all "
        "sit at zero, and the trained model, where all but L1 sit left of the dip.",
    )
    left, base = 70.0, fig.body_top + 290
    fig.bottom = base
    _phase_labels(fig, left, fig.body_top + 8)
    for row in trained:
        layer = row["layer"]
        start = (
            mean(s["median_h"] for s in fresh[layer]),
            mean(s["median_h_sd"] for s in fresh[layer]),
        )
        _ridge(fig, layer, start, (row["median_h"], row["median_h_sd"]), (left, base))
    for tick in (-3, -2, -1, 0, 1, 2):
        x = left + (tick - SPAN[0]) * SCALE
        fig.line(x, base, x, base + 4, "axis")
        fig.text(x, base + 18, number(tick, 0), "small", "middle")
    front = left + (dip - SPAN[0]) * SCALE
    fig.line(front, base, front + 11 * LAYER_SHIFT, base - 11 * LAYER_RISE, "dipfloor")
    fig.text(left + 280, base + 33, "pre-activation h", "small", "middle")
    return fig


def _surface_grid(rows: list[dict], condition: str) -> list[list[float]]:
    """Return the rest for each layer and step of one long run."""
    steps = [0, *LONG_STEPS]
    return [
        [
            _rest(_run(rows, condition, step=step, part="training_long")["layers"][layer])
            for step in steps
        ]
        for layer in range(4)
    ]


def _contour(a: tuple[Point, float], b: tuple[Point, float], level: float) -> Point | None:
    """Return the point on an edge where the value crosses ``level``, or None."""
    (point_a, value_a), (point_b, value_b) = a, b
    if (value_a - level) * (value_b - level) >= 0:
        return None
    share = (level - value_a) / (value_b - value_a)
    return (
        point_a[0] + share * (point_b[0] - point_a[0]),
        point_a[1] + share * (point_b[1] - point_a[1]),
    )


def _surface(fig: Figure, grid: list[list[float]], origin: Point, cls: str, dip: float) -> None:
    """Draw one surface, from the back row to the front, with a line where it crosses the dip."""

    def place(column: int, row: int, rest: float) -> Point:
        return (
            origin[0] + column * 34 + (3 - row) * 26,
            origin[1] - (3 - row) * 26 - rest * 66,
        )

    for row in range(3):
        for column in range(6):
            corners = [(column, row), (column + 1, row), (column + 1, row + 1), (column, row + 1)]
            values = [grid[r][c] for c, r in corners]
            points = [place(c, r, grid[r][c]) for c, r in corners]
            shade = min(0.95, 0.12 - mean(values) * 0.55)
            fig.path(points, "bg2", close=True)
            fig.shaded(points, cls, shade)
            fig.path([*points, points[0]], "mesh")
            pairs = [(points[i], values[i]) for i in range(4)]
            cross = [
                hit
                for i in range(4)
                if (hit := _contour(pairs[i], pairs[(i + 1) % 4], dip)) is not None
            ]
            if len(cross) == 2:
                fig.line(*cross[0], *cross[1], "contour")
    _surface_labels(fig, grid, place)


def _surface_labels(fig: Figure, grid: list[list[float]], place: Callable) -> None:
    """Label the surface's layers, steps and two far corners."""
    anchor_x, anchor_y = place(6, 0, 0)
    fig.text(anchor_x + 14, anchor_y + 22, "Step 12,000", "small")
    for row in range(4):
        fig.text(
            anchor_x + 14, anchor_y + 40 + row * 17, f"L{row}  {number(grid[row][6])}", "small"
        )
    for row in (0, 3):
        x, y = place(0, row, grid[row][0])
        fig.text(x - 6, y - 5, f"L{row}", "small", "end")
    for column, label in ((3, "6k"), (6, "12k")):
        x, y = place(column, 3, grid[3][column])
        fig.text(x, y + 20, label, "small", "middle")
    x, y = place(3, 3, grid[3][3])
    fig.text(x, y + 35, "training step", "small", "middle")


def surface(rows: list[dict], dip: float) -> Figure:
    """Draw the rest over layers and steps as two surfaces."""
    fig = Figure(
        ["Two routes dig nearly the same valley"],
        [
            "Height is the rest, by layer and step, on seed 0. Deeper is a stronger colour.",
            "The bold line marks where each surface crosses GELU's dip.",
        ],
        "Two three-dimensional surfaces of rest against layer and training step, for the baseline "
        "and the centred learned rest. Both fall steeply on L1 and L2, and less on L0 and L3.",
    )
    top = fig.body_top + 72
    fig.text(20, fig.body_top + 8, "Baseline: the weights set it", "head")
    fig.text(370, fig.body_top + 8, "Centred: a learned value sets it", "head")
    _surface(fig, _surface_grid(rows, "gelu"), (30, top + 60), "fc1", dip)
    _surface(fig, _surface_grid(rows, "rest_learned_fast"), (380, top + 60), "fc2", dip)
    fig.bottom = top + 158
    return fig
