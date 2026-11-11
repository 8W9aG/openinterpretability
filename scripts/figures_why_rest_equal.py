"""The equal-rest figure of papers/why-neurons-rest/README.md, from its results.jsonl."""

from figure_svg import Figure, Panel
from figures_why_rest_push import PUSH_LAYERS, push_ratio

# Each activation held at a fixed rest: label, the condition at each held rest, stroke,
# fill, and whether its dots are open. The colours are those of the push figure.
HELD_SERIES = [
    (
        "Softplus",
        {0: "softplus_rest_0", -1: "softplus_rest_-1", -2: "softplus_rest_-2"}
        | {-4: "softplus_rest_-4"},
        "sc3",
        "fc3",
        False,
    ),
    (
        "ReLU",
        {0: "relu_rest_0", -0.5: "relu_rest_-0.5", -1: "relu_rest_-1", -2: "relu_rest_-2"}
        | {-4: "relu_rest_-4"},
        "sc1",
        "fc1",
        False,
    ),
    ("Leaky 0.1", {0: "leaky_relu_rest_0", -1: "leaky_relu_rest_-1"}, "sc1", "fc1", True),
    ("GELU", {0: "rest_0", -0.5: "rest_-0.5", -1: "rest_-1"}, "sc2", "fc2", False),
    ("SiLU", {0: "silu_rest_0", -1: "silu_rest_-1"}, "sc2", "fc2", True),
    ("Tanh", {0: "tanh_rest_0"}, "sneutral", "fneutral", False),
]
HELD_PUSH_PARTS = ("push", "push_depth")  # the parts of the push rows of the held runs
# Where each label of the right panel goes from its deepest dot: dx, dy and anchor.
HELD_LABELS = {
    "Softplus": (0, -10, "middle"),
    "ReLU": (0, -10, "middle"),
    "Leaky 0.1": (-10, 4, "end"),
    "GELU": (-10, 4, "end"),
    "SiLU": (-10, 4, "end"),
    "Tanh": (0, -10, "middle"),
}
EQUAL_LEGEND = [
    ("fc3", "Softplus"),
    ("fc1", "ReLU and leaky ReLU 0.1 (open)"),
    ("fc2", "GELU and SiLU (open)"),
    ("fneutral", "Tanh"),
]


def held_rows(rows: list[dict], condition: str, seed: int = 0) -> list[dict]:
    """Return the push rows of one held run, in order of step."""
    found = [r for r in rows if r["part"] in HELD_PUSH_PARTS and r["condition"] == condition]
    return sorted((r for r in found if r["seed"] == seed), key=lambda r: r["step"])


def mean_push_ratio(rows: list[dict], condition: str, seed: int = 0) -> float | None:
    """Return the mean push ratio over a held run's steps, or None if it did not run."""
    found = held_rows(rows, condition, seed)
    return sum(push_ratio(r) for r in found) / len(found) if found else None


def _time_panel(panel: Panel, rows: list[dict]) -> None:
    """Draw the push ratio of each activation held at rest 0, over training, seed 0."""
    panel.grid_x([0, 1000, 2000, 3000, 4000], 0, thousands=True)
    panel.grid_y([-0.1, -0.05, 0, 0.05], 2)
    panel.horizontal(0, "axis")
    panel.x_title("step")
    low, high = panel.y_range
    for _, conditions, stroke, _, _ in HELD_SERIES:
        found = held_rows(rows, conditions[0])
        steps = [r["step"] for r in found]
        ratios = [min(max(push_ratio(r), low), high) for r in found]
        panel.curve(steps, ratios, f"line {stroke}")


def _depth_panel(fig: Figure, panel: Panel, rows: list[dict]) -> None:
    """Draw the mean push ratio against the rest that each run is held at, seed 0."""
    panel.grid_x([-4, -3, -2, -1, 0], 0)
    panel.grid_y([-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04], 2)
    panel.horizontal(0, "axis")
    panel.x_title("held rest")
    for label, conditions, stroke, fill, open_dots in HELD_SERIES:
        points = []
        for rest in sorted(conditions):
            ratio = mean_push_ratio(rows, conditions[rest])
            if ratio is not None:
                points.append((rest, ratio))
        if len(points) > 1:
            panel.curve([p[0] for p in points], [p[1] for p in points], f"line {stroke}")
        for rest, ratio in points:
            cls = f"open {stroke}" if open_dots else f"{fill} ring"
            fig.dot(panel.x(rest), panel.y(ratio), 4.5, cls)
        if points:
            dx, dy, anchor = HELD_LABELS[label]
            rest, ratio = points[0]
            fig.text(panel.x(rest) + dx, panel.y(ratio) + dy, label, "small", anchor)


def equal_rest(rows: list[dict]) -> Figure:
    """Draw the push at an equal rest of 0 over training, and the push at each held rest."""
    fig = Figure(
        ["Held at the same rest, the push depends on the depth"],
        [
            "Push ratio of runs whose h is centred and held at a fixed rest, median over neurons, "
            "mean over",
            f"L{PUSH_LAYERS[0]}–L{PUSH_LAYERS[-1]}, seed 0. Left: held at 0. Right: the mean "
            "over steps 1–4,000 at each held rest.",
        ],
        "Left, lines of the push ratio over 4,000 steps with every activation held at rest 0. "
        "Softplus, ReLU, leaky ReLU 0.1 and GELU fall together to about minus 0.06 to minus "
        "0.07. SiLU is weaker, near minus 0.04 to minus 0.05, and tanh stays at 0. GELU and "
        "SiLU start above 0 for the first 100 steps. Right, the mean push ratio against the "
        "held rest: softplus stays between minus 0.07 and minus 0.05 from 0 to minus 4. ReLU "
        "falls from minus 0.07 at 0 to minus 0.04 at minus 0.5, minus 0.01 at minus 1, and 0 "
        "at minus 2 and minus 4. Leaky ReLU 0.1 turns to plus 0.03 at minus 1. GELU is minus "
        "0.03 at minus 0.5 and minus 0.04 at minus 1. SiLU falls to minus 0.02 at minus 1.",
    )
    fig.legend(fig.body_top + 10, EQUAL_LEGEND)
    top = fig.body_top + 44
    left = Panel(fig, (56, top, 330, 230), (0, 4000), (-0.12, 0.06))
    right = Panel(fig, (460, top, 270, 230), (-4.3, 0.3), (-0.09, 0.045))
    left.head("Push ratio at rest 0")
    right.head("Mean push against the held rest")
    _time_panel(left, rows)
    _depth_panel(fig, right, rows)
    return fig
