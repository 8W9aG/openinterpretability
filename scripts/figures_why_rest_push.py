"""The push figure of papers/why-neurons-rest/README.md, from its results.jsonl."""

from figure_svg import Figure, Panel
from figures_why_rest import _rest, _run

# Each activation of the push figure: label, condition, stroke, fill. The first colour
# marks a push down from the first steps, the second a push that points up at first.
PUSH_SERIES = [
    ("Softplus", "softplus", "sc3", "fc3"),
    ("ReLU", "relu", "sc1", "fc1"),
    ("Leaky 0.01", "leaky_relu_0.01", "sc1", "fc1"),
    ("Leaky 0.1", "leaky_relu", "sc1", "fc1"),
    ("GELU", "gelu", "sc2", "fc2"),
    ("SiLU", "silu", "sc2", "fc2"),
    ("Tanh", "tanh", "sneutral", "fneutral"),
]
PUSH_LAYERS = (1, 2, 3)
PUSH_CLUSTER = ("relu", "gelu", "leaky_relu_0.01")  # these rest within 0.05 of each other
# Where each label of the right panel goes from its seed-0 dot: dx, dy and anchor.
PUSH_LABELS = {
    "softplus": (9, 4, "start"),
    "leaky_relu": (-10, -8, "end"),
    "silu": (9, 4, "start"),
    "tanh": (-10, -8, "end"),
}
PUSH_LEGEND = [
    ("fc3", "Softplus"),
    ("fc1", "ReLU and leaky ReLU"),
    ("fc2", "GELU and SiLU"),
    ("fneutral", "Tanh"),
]


def push_ratio(row: dict) -> float:
    """Return the mean over L1–L3 of the median push ratio in one push row."""
    return sum(row["layers"][i]["median_push_ratio"] for i in PUSH_LAYERS) / len(PUSH_LAYERS)


def push_rows(rows: list[dict], condition: str, seed: int) -> list[dict]:
    """Return the push rows of one run, in order of step."""
    found = [r for r in rows if r["part"] == "push" and r["condition"] == condition]
    return sorted((r for r in found if r["seed"] == seed), key=lambda r: r["step"])


def push_run(rows: list[dict], condition: str, seed: int) -> dict | None:
    """Return the step-4,000 row of the run that recorded the push, or None."""
    return _run(rows, condition, seed, part="training_push_rerun") or _run(rows, condition, seed)


def _push_time_panel(fig: Figure, panel: Panel, rows: list[dict]) -> None:
    """Draw the push ratio of each activation over training, seed 0."""
    panel.grid_x([0, 1000, 2000, 3000, 4000], 0, thousands=True)
    panel.grid_y([-0.1, -0.05, 0, 0.05], 2)
    panel.horizontal(0, "axis")
    panel.x_title("step")
    for _, condition, stroke, _ in PUSH_SERIES:
        found = push_rows(rows, condition, 0)
        steps = [r["step"] for r in found]
        ratios = [max(push_ratio(r), panel.y_range[0]) for r in found]
        panel.curve(steps, ratios, f"line {stroke}")


def _push_depth_panel(fig: Figure, panel: Panel, rows: list[dict]) -> None:
    """Draw L1's rest at step 4,000 against the run's mean push ratio, one dot for each seed."""
    panel.grid_x([-0.04, -0.03, -0.02, -0.01, 0], 2)
    panel.grid_y([-5, -4, -3, -2, -1, 0], 0)
    panel.horizontal(0, "axis")
    panel.vertical(0, "axis")
    panel.x_title("push ratio, mean over steps 1–4,000")
    for label, condition, _, fill in PUSH_SERIES:
        points = []
        for seed in (0, 1):
            found, run = push_rows(rows, condition, seed), push_run(rows, condition, seed)
            if not found or run is None:
                continue
            ratio = sum(push_ratio(r) for r in found) / len(found)
            points.append((panel.x(ratio), panel.y(_rest(run["layers"][1]))))
            fig.dot(*points[-1], 5, f"{fill} ring")
        x, y = points[0]
        if condition == PUSH_CLUSTER[0]:
            fig.text(x - 10, y + 4, "ReLU, GELU, leaky 0.01", "small", "end")
        elif condition in PUSH_LABELS:
            dx, dy, anchor = PUSH_LABELS[condition]
            fig.text(x + dx, y + dy, label, "small", anchor)


def push(rows: list[dict]) -> Figure:
    """Draw the push on the rest over training, and its mean against the depth it reaches."""
    fig = Figure(
        ["A steadier push down gives a deeper rest"],
        [
            "Push ratio: the mean over 100 steps of −dL/dh on each neuron's whole h, over its RMS, "
            "median over",
            "neurons, mean over L1–L3. Left: seed 0. Right: L1's median h at step 4,000, one dot "
            "for each seed.",
        ],
        "Left, lines of the push ratio over 4,000 steps for seven activations. Softplus stays "
        "lowest, near minus 0.04 to minus 0.1. ReLU and both leaky ReLUs are below 0 from the "
        "start. GELU and SiLU start above 0 and cross below it after a few hundred steps. Tanh "
        "stays at 0. Right, L1's rest against the mean push ratio: softplus near minus 0.043 "
        "and minus 4.7 to minus 5, ReLU, GELU and leaky 0.01 near minus 0.02 and minus 1.1, "
        "leaky 0.1 near minus 0.016 and minus 0.7, SiLU near minus 0.01 and minus 0.5, and tanh "
        "at 0 and 0.",
    )
    fig.legend(fig.body_top + 10, PUSH_LEGEND)
    top = fig.body_top + 44
    left = Panel(fig, (56, top, 330, 230), (0, 4000), (-0.12, 0.06))
    right = Panel(fig, (460, top, 270, 230), (-0.05, 0.005), (-5.3, 0.3))
    left.head("Push ratio over training")
    right.head("Rest against mean push")
    _push_time_panel(fig, left, rows)
    _push_depth_panel(fig, right, rows)
    return fig
