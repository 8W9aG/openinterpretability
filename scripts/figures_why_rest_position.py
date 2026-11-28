"""The position figure of papers/why-neurons-rest/README.md, from its results.jsonl (#7202)."""

from figure_svg import Figure, Panel

LAYER = 1  # the layer whose rest the long window deepens
BUCKET = "16-127"  # the positions that the rest is read at, in every model
PYTHIA = "EleutherAI/pythia-70m"
# Each run: label, condition (None for Pythia-70m), stroke and fill.
SERIES = [
    ("Window of 2,048, learned positions", "pythia_size_batch_32x_window", "sc1", "fc1"),
    ("Window of 2,048, rotary positions", "pythia_size_batch_32x_window_rotary", "sc2", "fc2"),
    ("Window of 128, learned positions", "pythia_size_batch_32x", "sneutral", "fneutral"),
    ("Pythia-70m", None, "sc3", "fc3"),
]
LAST_STEP = 2000


def position_rows(rows: list[dict], condition: str | None) -> dict[object, list[dict]]:
    """The ``rest_by_position`` rows of one run, by seed, in order of step.

    Args:
        rows: The paper's results.
        condition: The run's condition, or None for Pythia-70m.

    Returns:
        The rows of each seed, up to ``LAST_STEP``. Pythia-70m's seed is None.

    Examples:
        >>> rows = [{"part": "rest_by_position", "condition": "a", "seed": 0, "step": 9},
        ...         {"part": "rest_by_position", "condition": "a", "seed": 0, "step": 3},
        ...         {"part": "rest_by_position", "model": PYTHIA, "step": 1}]
        >>> [r["step"] for r in position_rows(rows, "a")[0]], list(position_rows(rows, None))
        ([3, 9], [None])
    """
    found: dict[object, list[dict]] = {}
    for r in rows:
        if r.get("part") != "rest_by_position" or r["step"] > LAST_STEP:
            continue
        if (r.get("condition") == condition) if condition else (r.get("model") == PYTHIA):
            found.setdefault(r.get("seed"), []).append(r)
    return {seed: sorted(found[seed], key=lambda r: r["step"]) for seed in sorted(found, key=str)}


def _panel(fig: Figure, panel: Panel, rows: list[dict], read) -> None:
    """Draw one measure of layer ``LAYER`` over training, one line a run and seed.

    Args:
        fig: The figure.
        panel: The panel.
        rows: The paper's results.
        read: The measure, from one layer's statistics.
    """
    panel.grid_x([0, 1000, 2000], 0, thousands=True)
    panel.x_title("step")
    for _, condition, stroke, fill in SERIES:
        for found in position_rows(rows, condition).values():
            steps = [r["step"] for r in found]
            values = [read(r["layers"][LAYER]) for r in found]
            panel.curve(steps, values, f"line {stroke}")
            for step, value in zip(steps, values, strict=True):
                fig.dot(panel.x(step), panel.y(value), 3.5, f"{fill} ring")


def _listed(values) -> str:
    """The values to two places, joined with commas.

    Args:
        values: The numbers.

    Returns:
        The text.

    Examples:
        >>> _listed([0.5, -4.123])
        '0.50, -4.12'
    """
    return ", ".join(f"{v:.2f}" for v in values)


def position(rows: list[dict]) -> Figure:
    """Draw L1's position share and rest over training, in each run and in Pythia-70m.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    ends = {label: [found[-1]["layers"][LAYER] for found in position_rows(rows, c).values()]
            for label, c, _, _ in SERIES}  # fmt: skip
    said = "; ".join(
        f"{label}: position share {_listed(e['position_share'] for e in end)}, "
        f"rest {_listed(e['rest'][BUCKET] for e in end)}"
        for label, end in ends.items()
        if end
    )
    fig = Figure(
        [
            "With learned positions, a long window fills L1's input with position,",
            "and L1's neurons rest deep",
        ],
        [
            f"L{LAYER}, each run and seed. Left: the share of the MLP input's energy that is its "
            "position's mean over windows.",
            f"Right: the rest at positions {BUCKET.replace('-', '–')}, the median over neurons "
            "of each neuron's median h.",
        ],
        f"Two panels over steps 0 to {LAST_STEP:,}. At the last step, {said}.",
    )
    entries = [(fill, label) for label, _, _, fill in SERIES]
    fig.legend(fig.body_top + 10, entries[:2])  # two rows, so the labels fit
    fig.legend(fig.body_top + 30, entries[2:])
    top = fig.body_top + 68
    left = Panel(fig, (56, top, 300, 220), (0, LAST_STEP), (0, 1))
    right = Panel(fig, (430, top, 300, 220), (0, LAST_STEP), (-5, 0.2))
    left.head("Position share of the input")
    right.head("Rest")
    left.grid_y([0, 0.25, 0.5, 0.75, 1], 2)
    right.grid_y([-5, -4, -3, -2, -1, 0], 0)
    right.horizontal(0, "axis")
    _panel(fig, left, rows, lambda layer: layer["position_share"])
    _panel(fig, right, rows, lambda layer: layer["rest"][BUCKET])
    return fig
