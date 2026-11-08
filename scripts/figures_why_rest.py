"""The figures of papers/why-neurons-rest/README.md, from its results.jsonl."""

import math

from figure_svg import (
    Figure,
    Panel,
    columns,
    gelu,
    leaky_relu,
    number,
    relu,
    sample,
    silu,
    softplus,
)

STEP = 4000
EVAL_STEPS = list(range(0, STEP + 1, 500))
CYCLE = "9s"
LONG_STEPS = [2000, 4000, 6000, 8000, 10000, 12000]
EFFECT_ROWS = [
    ("Baseline (GELU)", "gelu"),
    ("1× width", "mlp_1x"),
    ("16× width", "mlp_16x"),
    ("No dropout", "dropout_0"),
    ("Dropout 0.3", "dropout_0.3"),
    ("Weight decay 0.1", "weight_decay_0.1"),
    ("ReLU", "relu"),
    ("Tanh", "tanh"),
]
# Each series of the rate figure: label, GELU condition, ReLU condition, stroke, fill.
RATE_SERIES = [
    ("AdamW 1e-3", "lr_1e-3", "relu_lr_1e-3", "sc1", "fc1"),
    ("AdamW 3e-4", "gelu", "relu", "sc1", "fc1"),
    ("AdamW 1e-4", "lr_1e-4", "relu_lr_1e-4", "sc1", "fc1"),
    ("SGD 0.1", "sgd", "relu_sgd", "sneutral", "fneutral"),
]
LOSS_ROWS = [
    ("ReLU", "relu", "fc1", "sc1"),
    ("Baseline (GELU)", "gelu", "fc1", "sc1"),
    ("Centred, learned, 30× rate", "rest_learned_fast", "fc2", "sc2"),
    ("Centred, learned, base rate", "rest_learned", "fc2", "sc2"),
    ("Centred, rest 0", "rest_0", "fneutral", "sneutral"),
    ("Centred, rest −1", "rest_-1", "fneutral", "sneutral"),
    ("Tanh", "tanh", "fc1", "sc1"),
]


def _rest(layer: dict) -> float:
    """Return the rest of one layer's row: the learned value if it has one, or the median h."""
    learned = layer.get("median_learned_rest")
    return layer["median_h"] if learned is None else learned


def _run(rows: list[dict], condition: str, seed: int = 0, step: int = STEP, part="training"):
    """Return the row of one condition at one step, or None if that seed did not run."""
    for row in rows:
        same_run = row.get("condition") == condition and row.get("seed") == seed
        if same_run and row["part"] == part and row["step"] == step:
            return row
    return None


def _moving_dots(fig: Figure, panel: Panel, rows: list[dict], condition: str, shape) -> None:
    """Add a dot for each of L1–L3 that follows the measured rest over the training steps."""
    for layer in (1, 2, 3):
        frames = []
        for step in EVAL_STEPS:
            depth = _rest(_run(rows, condition, step=step)["layers"][layer])
            share = 70 * step / STEP
            frames.append(
                f"{share:.1f}%{{transform:translate({panel.x(depth):.1f}px,"
                f"{panel.y(shape(depth)):.1f}px)}}"
            )
        name = f"{condition}{layer}"
        last = frames[-1].split("{", 1)[1].rstrip("}")
        fig.motion += (
            f"@keyframes {name}{{{''.join(frames)}100%{{{last}}}}}"
            f".{name}{{animation:{name} {CYCLE} linear infinite;{last}}}"
        )
        fig.dot(0, 0, 5, f"fc1 ring {name}")


def _progress(fig: Figure, y: float) -> None:
    """Add a bar that shows the training step of the moving dots."""
    fig.text(44, y + 4, "step 0", "small")
    fig.line(90, y, 660, y, "axis")
    fig.text(668, y + 4, f"{STEP:,}", "small")
    fig.motion += (
        f"@keyframes progress{{0%{{transform:translateX(90px)}}70%{{transform:translateX(660px)}}"
        f"100%{{transform:translateX(660px)}}}}"
        f".progress{{animation:progress {CYCLE} linear infinite;transform:translateX(660px)}}"
    )
    fig.marks.append(f'<circle class="fc1 progress" cx="0" cy="{y:.1f}" r="4"/>')


def _lowest(shape, low: float, high: float) -> float:
    """Return where ``shape`` has its minimum between ``low`` and ``high``, by golden section."""
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = high - ratio * (high - low), low + ratio * (high - low)
        low, high = (low, b) if shape(a) < shape(b) else (a, high)
    return (low + high) / 2


def _activation_panels(fig: Figure, rows: list[dict], shapes, x_range, x_ticks) -> None:
    """Draw one panel for each activation, with its rest at L1–L3 moving over the steps.

    ``shapes`` holds (name, condition, function, has_dip) for each panel. A dip is marked
    where the function has a minimum below 0.
    """
    panels = columns(fig, len(shapes), 44, fig.body_top + 22, 200, x_range, (-1.15, 2.2))
    for panel, (name, condition, shape, has_dip) in zip(panels, shapes, strict=True):
        panel.head(name)
        panel.grid_x(x_ticks, 0)
        panel.grid_y([-1, 0, 1, 2], 0, labels=panel is panels[0])
        panel.horizontal(0, "axis")
        panel.vertical(0, "axis")
        xs = sample(*x_range)
        panel.curve(xs, [shape(x) for x in xs], "curve")
        panel.x_title("pre-activation h")
        fig.dot(panel.x(0), panel.y(shape(0)), 5, "open sc1")
        depths = [_rest(layer) for layer in _run(rows, condition)["layers"][1:]]
        _moving_dots(fig, panel, rows, condition, shape)
        low, high = min(depths), max(depths)
        reading = f"{number(low)} to {number(high)}" if high - low > 0.01 else "all at 0.00"
        fig.text(panel.left + panel.width, panel.y(-0.72), reading, "small", "end")
        if has_dip:
            dip = _lowest(shape, x_range[0], 0)
            fig.line(panel.x(dip), panel.y(shape(dip)), panel.x(dip), panel.y(0.75), "dip")
            label = f"dip at {number(dip)}"
            fig.text(panel.x(dip), panel.y(0.75) - 5, label, "small", "middle")
    _progress(fig, panels[0].top + panels[0].height + 48)
    fig.bottom += 26


def activations(rows: list[dict]) -> Figure:
    """Draw GELU, ReLU and tanh with the place where their neurons rest at step 4,000."""
    fig = Figure(
        ["Neurons rest low under one-sided activations, with or without a dip"],
        [
            "Each moving dot is one layer's median neuron (L1–L3), measured every 500 steps on "
            "small models.",
            "Training starts at h ≈ 0 (open circle). ReLU has no dip, and its neurons go as "
            "deep as GELU's.",
        ],
        "Three plots of an activation function against pre-activation h. Under GELU and ReLU, "
        "layers L1 to L3 rest between about minus 0.8 and minus 1.2. Under tanh they stay at 0.",
    )
    shapes = [
        ("GELU", "gelu", gelu, True),
        ("ReLU", "relu", relu, False),
        ("Tanh", "tanh", math.tanh, False),
    ]
    _activation_panels(fig, rows, shapes, (-2.5, 2.5), [-2, -1, 0, 1, 2])
    return fig


def more_activations(rows: list[dict]) -> Figure:
    """Draw softplus, leaky ReLU and SiLU with the place where their neurons rest."""
    fig = Figure(
        ["Every one-sided activation rests below 0, but neither a dip nor a floor sets the depth"],
        [
            "Each moving dot is one layer's median neuron (L1–L3), every 500 steps. Softplus has "
            "no dip and goes deepest.",
            "SiLU's dip is deeper than GELU's, yet it rests shallower. Leaky ReLU passes 10% "
            "below 0 and rests shallower than ReLU.",
        ],
        "Three plots of an activation function against pre-activation h, from minus 5 to 2. "
        "Under softplus, layers L1 to L3 rest at about minus 3.5 to minus 4.7. Under leaky ReLU "
        "they rest at about minus 0.7 to minus 0.8, and under SiLU at about minus 0.2 to "
        "minus 0.5.",
    )
    shapes = [
        ("Softplus", "softplus", softplus, False),
        ("Leaky ReLU, slope 0.1", "leaky_relu", leaky_relu, False),
        ("SiLU", "silu", silu, True),
    ]
    _activation_panels(fig, rows, shapes, (-5, 2), [-4, -2, 0, 2])
    return fig


def _spread(ys: list[float], gap: float) -> list[float]:
    """Return the label heights ``ys``, moved down where needed so that each is ``gap`` apart."""
    order = sorted(range(len(ys)), key=ys.__getitem__)
    out = list(ys)
    for before, index in zip(order, order[1:], strict=False):
        out[index] = max(out[index], out[before] + gap)
    return out


def _rate_panel(fig: Figure, panel: Panel, rows: list[dict], relu_side: bool) -> None:
    """Draw L1's rest against the held-out loss over training, one line for each series."""
    panel.grid_x([7, 6.5, 6, 5.5], 1)
    panel.grid_y([-4, -3, -2, -1, 0], 0, labels=not relu_side)
    panel.horizontal(0, "axis")
    panel.x_title("held-out loss (training goes right)")
    ends = []
    for label, gelu_name, relu_name, stroke, fill in RATE_SERIES:
        condition = relu_name if relu_side else gelu_name
        points = [_run(rows, condition, step=step) for step in EVAL_STEPS[1:]]
        losses = [row["loss"] for row in points]
        rests = [_rest(row["layers"][1]) for row in points]
        panel.curve(losses, rests, f"line {stroke}")
        fig.dot(panel.x(losses[-1]), panel.y(rests[-1]), 4, f"{fill} ring")
        ends.append((label, panel.x(losses[-1]), panel.y(rests[-1])))
    heights = _spread([y + 4 for _, _, y in ends], 14)
    for (label, x, _), y in zip(ends, heights, strict=True):
        fig.text(x + 7, y, label, "small")


def rates(rows: list[dict]) -> Figure:
    """Draw how deep L1 rests for each optimizer and rate, against how far training has got."""
    fig = Figure(
        ["At the same loss, the rest is as deep as the optimizer's steps take it"],
        [
            "L1's median h against held-out loss, every 500 steps to step 4,000. Seed 0. A larger "
            "AdamW rate reaches the same",
            "loss with a deeper rest. Under SGD, GELU and ReLU part.",
        ],
        "Two line plots of layer 1's rest against loss, for GELU and ReLU. At AdamW rates of "
        "1e-4, 3e-4 and 1e-3, the rest at step 4,000 is about minus 0.4, minus 1.1 and minus 3 "
        "to minus 4. SGD rests at minus 0.32 under GELU and minus 0.59 under ReLU.",
    )
    top = fig.body_top + 22
    panels = columns(fig, 2, 48, top, 220, (7.1, 5.3), (-4.5, 0.2), gap=40, right=96)
    for panel, (name, relu_side) in zip(panels, (("GELU", False), ("ReLU", True)), strict=True):
        panel.head(name)
        _rate_panel(fig, panel, rows, relu_side)
    return fig


def _effect_panel(fig: Figure, panel: Panel, rows: list[dict], layer: int, dip: float) -> None:
    """Draw one layer's rest for each condition, against the baseline's line."""
    fig.text(panel.left + panel.width, panel.top - 8, f"L{layer}", "head", "end")
    panel.grid_x([-1, -0.5, 0], 1)
    panel.vertical(dip, "dip")
    fig.text(panel.x(dip), panel.top - 8, "dip", "small", "middle")
    panel.vertical(_rest(_run(rows, "gelu")["layers"][layer]), "line sc1 soft")
    panel.x_title("median h")
    for index, (_, condition) in enumerate(EFFECT_ROWS):
        y = panel.top + (index + 0.5) * panel.height / len(EFFECT_ROWS)
        second = _run(rows, condition, seed=1)
        if second:
            fig.dot(panel.x(_rest(second["layers"][layer])), y, 4, "open sc1")
        first = _rest(_run(rows, condition)["layers"][layer])
        fig.dot(panel.x(first), y, 5, "fc1 ring")


def effects(rows: list[dict], dip: float) -> Figure:
    """Draw the rest of each layer under each change to the baseline."""
    fig = Figure(
        ["Width and regularisation barely move the rest, and only tanh removes it"],
        [
            "Median h at step 4,000, each panel one layer. The pale blue line is the "
            "baseline. The grey line is the",
            "dip. L0 is the exception: it reads the embeddings, and it deepens with dropout "
            "on this one seed.",
        ],
        "Dot plots of median pre-activation by condition for layers 0 to 3. At L1 to L3, width, "
        "dropout, weight decay and ReLU stay close to the baseline, and tanh sits at 0.",
    )
    top = fig.body_top + 44
    panels = columns(fig, 4, 136, top, 232, (-1.4, 0.25), (0, 1), gap=14)
    fig.legend(fig.body_top + 10, [("fc1", "Seed 0"), ("open sc1", "Seed 1, where it ran")])
    for index, (label, _) in enumerate(EFFECT_ROWS):
        y = top + (index + 0.5) * 232 / len(EFFECT_ROWS)
        fig.text(128, y + 4, label, "ink" if index == 0 else "", "end")
    for layer, panel in enumerate(panels):
        _effect_panel(fig, panel, rows, layer, dip)
    return fig


def _long_panel(fig: Figure, panel: Panel, rows: list[dict], layer: int, dip: float) -> None:
    """Draw one layer's rest over steps for the two routes."""
    panel.head(f"L{layer}")
    panel.grid_x([0, 6000, 12000], 0, thousands=True)
    panel.grid_y([-1.5, -1, -0.5, 0], 1, labels=layer == 0)
    panel.horizontal(dip, "dip")
    steps = [0, *LONG_STEPS]
    series = [("gelu", "line sc1", "fc1"), ("rest_learned_fast", "line sc2", "fc2")]
    fig.marks.append(
        f'<clipPath id="reveal{layer}"><rect class="reveal" x="{panel.left - 6:.1f}" '
        f'y="{panel.top - 6:.1f}" width="{panel.width + 14:.1f}" height="{panel.height + 12:.1f}"/>'
        f'</clipPath><g clip-path="url(#reveal{layer})">'
    )
    ends = []
    for condition, line, fill in series:
        rest = [
            _rest(_run(rows, condition, step=step, part="training_long")["layers"][layer])
            for step in steps
        ]
        panel.curve(steps, rest, line)
        fig.dot(panel.x(steps[-1]), panel.y(rest[-1]), 4, f"{fill} ring")
        ends.append(rest[-1])
    if abs(ends[0] - ends[1]) > 0.07:
        for value, shift in zip(ends, (15, -9), strict=True):
            fig.text(panel.x(steps[-1]), panel.y(value) + shift, number(value), "small", "end")
    fig.marks.append("</g>")


def long_run(rows: list[dict], dip: float) -> Figure:
    """Draw the rest over 12,000 steps for the baseline and the fast control."""
    fig = Figure(
        ["Two routes reach the rest at nearly the same pace, and −1 is not where it stops"],
        [
            "With centring, the weights can't set the rest, so a learned value for each "
            "neuron does it. Seed 0.",
        ],
        "Line plots of rest against training step for layers 0 to 3. The baseline and the centred "
        "learned rest fall together. At L1 they reach minus 1.46 and minus 1.32 by step 12,000.",
    )
    fig.legend(
        fig.body_top + 10,
        [("fc1", "Baseline: the weights set it"), ("fc2", "Centred: a learned value sets it")],
    )
    top = fig.body_top + 46
    panels = columns(fig, 4, 48, top, 196, (0, 12000), (-1.65, 0.1), gap=22)
    for layer, panel in enumerate(panels):
        _long_panel(fig, panel, rows, layer, dip)
        panel.x_title("training step")
    fig.text(panels[0].left + 4, panels[0].y(dip) - 5, "dip", "small")
    fig.motion += (
        "@keyframes reveal{0%{transform:scaleX(0)}70%,100%{transform:scaleX(1)}}"
        f".reveal{{transform-box:fill-box;transform-origin:left;"
        f"animation:reveal {CYCLE} linear infinite}}"
    )
    return fig


def _loss_rows(fig: Figure, panel: Panel, rows: list[dict]) -> None:
    """Draw the loss of each condition at step 4,000, one dot for each seed."""
    panel.grid_x([5.5, 5.6, 5.7], 1)
    panel.x_title("loss at step 4,000 (lower is better)")
    gap = panel.height / len(LOSS_ROWS)
    for index, (label, condition, fill, stroke) in enumerate(LOSS_ROWS):
        y = panel.top + (index + 0.5) * gap
        fig.text(panel.left - 10, y + 4, label, "", "end")
        second = _run(rows, condition, seed=1)
        if second:
            fig.dot(panel.x(second["loss"]), y, 4, f"open {stroke}")
        fig.dot(panel.x(_run(rows, condition)["loss"]), y, 5, f"{fill} ring")


def _gap_panel(fig: Figure, panel: Panel, rows: list[dict]) -> None:
    """Draw how far the fast control's loss stays above the baseline's."""
    gaps = [
        _run(rows, "rest_learned_fast", step=step, part="training_long")["loss"]
        - _run(rows, "gelu", step=step, part="training_long")["loss"]
        for step in LONG_STEPS
    ]
    panel.grid_x([2000, 7000, 12000], 0, thousands=True)
    panel.grid_y([0, 0.05, 0.1], 2)
    panel.horizontal(0, "axis")
    panel.curve(LONG_STEPS, gaps, "line sc2")
    for step, gap in zip(LONG_STEPS, gaps, strict=True):
        fig.dot(panel.x(step), panel.y(gap), 4, "fc2 ring")
    fig.text(panel.x(LONG_STEPS[-1]), panel.y(gaps[-1]) - 9, number(gaps[-1], 3), "small", "end")
    fig.text(panel.x(LONG_STEPS[1]), panel.y(gaps[1]) + 18, number(gaps[1], 3), "small", "middle")
    panel.x_title("training step")


def loss(rows: list[dict]) -> Figure:
    """Draw the loss of each condition, and the gap that longer training doesn't close."""
    fig = Figure(
        ["A learned rest beats a forced one, but a gap to the baseline stays"],
        [
            "A forced rest of −1 trains as badly as a forced rest of 0. Longer training "
            "doesn't close the gap.",
        ],
        "Left: loss at step 4,000. ReLU and GELU are lowest, the learned rest at 30 times the "
        "rate is next, and forced rests and tanh are worst. Right: the gap stays between "
        "0.064 and 0.076.",
    )
    fig.legend(
        fig.body_top + 10,
        [
            ("fc1", "Weights set the rest"),
            ("fc2", "Learned value"),
            ("fneutral", "Forced"),
            ("open sneutral", "Seed 1, where it ran"),
        ],
    )
    top = fig.body_top + 44
    left = Panel(fig, (200, top, 300, 200), (5.46, 5.72), (0, 1))
    left.head("All conditions")
    _loss_rows(fig, left, rows)
    right = Panel(fig, (566, top, 170, 200), (2000, 12000), (0, 0.1))
    right.head("Fast control minus baseline")
    _gap_panel(fig, right, rows)
    return fig


def build(rows: list[dict], dip: float) -> dict[str, Figure]:
    """Return this analysis's figures by file name."""
    return {
        "activations.svg": activations(rows),
        "more-activations.svg": more_activations(rows),
        "rates.svg": rates(rows),
        "effects.svg": effects(rows, dip),
        "long-run.svg": long_run(rows, dip),
        "loss.svg": loss(rows),
    }
