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


def _push_time_panel(panel: Panel, rows: list[dict]) -> None:
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
    _push_time_panel(left, rows)
    _push_depth_panel(fig, right, rows)
    return fig


# The norm-pair conditions of #6465, top to bottom: final only, one block norm added, every norm.
NORM_PAIRS = (
    [("final only", "gelu_final_norm_centred_dropout_0")]
    + [
        (f"final + L{i} ln_{j}", f"gelu_final_L{i}_ln{j}_dropout_0")
        for i in range(4)
        for j in (1, 2)
    ]
    + [("every norm", "gelu_all_norms_centred_dropout_0")]
)
BELOW = 3  # the bins below h = -0.25, of share_output_bins_up's seven


def up_below(rows: list[dict], condition: str, seed: int) -> float:
    """The share of neurons whose output's push points up below h = −0.25, as Table 23.

    It is a mean over the three bins below −0.25, L1–L3 and the windows of steps 201–1,000.

    Args:
        rows: The paper's results.
        condition: The condition.
        seed: The seed.

    Returns:
        The share.
    """
    shares = [
        share
        for r in rows
        if r.get("part") == "push_norm_pairs" and r["condition"] == condition
        and r["seed"] == seed and r["step"] - r["steps"] + 1 >= 201 and r["step"] <= 1000
        for layer in (1, 2, 3)
        for share in r["layers"][layer]["share_output_bins_up"][:BELOW]
    ]  # fmt: skip
    return sum(shares) / len(shares)


def mean(values: list[float]) -> float:
    """The mean of ``values``.

    Args:
        values: The values.

    Returns:
        Their mean.
    """
    return sum(values) / len(values)


def norm_pairs(rows: list[dict]) -> Figure:
    """Each norm-pair condition's up share below the rest, for each seed (#6465).

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    found = {c: [up_below(rows, c, s) for s in (0, 1)] for _, c in NORM_PAIRS}
    one = [v for _, c in NORM_PAIRS[1:-1] for v in found[c]]  # one block norm added
    low, high = min(one), max(one)
    fig = Figure(
        ["No single block norm restores the up push", "that centring every norm keeps"],
        [
            "Each mark is one seed's share of GELU neurons whose output's push points up below "
            "h = −0.25.",
            "The final LayerNorm is centred alone, with one block norm, or with every norm, "
            "with no dropout.",
            "Circles: seed 0. Triangles: seed 1.",
        ],
        f"A dot plot of {len(NORM_PAIRS)} conditions. With only the final norm centred, "
        f"{mean(found[NORM_PAIRS[0][1]]):.2f} of neurons have an up push below the rest. "
        f"Adding any one block norm leaves each seed between {low:.2f} and {high:.2f}. "
        f"Centring every norm gives {mean(found[NORM_PAIRS[-1][1]]):.2f}.",
    )
    panel = Panel(fig, (150, fig.body_top + 20, 470, 24 * len(NORM_PAIRS)), (0.4, 0.9),
                  (0, len(NORM_PAIRS)))  # fmt: skip
    panel.grid_x([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 1)
    panel.x_title("share of neurons with an up push below h = −0.25")
    for i, (label, condition) in enumerate(NORM_PAIRS):
        y = panel.y(len(NORM_PAIRS) - i - 0.5)
        fig.text(panel.left - 10, y + 4, label, "small", "end")
        x0, x1 = (panel.x(v) for v in found[condition])
        fig.dot(x0, y, 4, "fc2")
        fig.path([(x1, y - 5), (x1 + 5, y + 4), (x1 - 5, y + 4)], "fc1", close=True)
    fig.bottom = panel.top + panel.height + 44
    return fig
