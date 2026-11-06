"""The figures of papers/default-on-poles/README.md, from its results.jsonl."""

from statistics import correlation, mean

from figure_svg import Figure, Panel, columns, gelu, normal, number, sample

GPT2 = "openai-community/gpt2"
QWEN = "Qwen/Qwen2.5-0.5B"
CONCEPT_LAYERS = [0, 3, 6, 9, 11]
COST_ROWS = [
    ("Negative side, shuffled", "null_negative", "fneutral"),
    ("Negative side, clamped", "clamp_negative", "fc1"),
    ("Positive side, shuffled", "null_positive", "fneutral"),
    ("Positive side, clamped", "clamp_positive", "fc2"),
    ("Negative side, zero", "zero_negative", "fc1"),
    ("Positive side, zero", "zero_positive", "fc2"),
    ("Every neuron at its mean", "mean", "fc3"),
]
ILLUSTRATION = [
    0.1, -0.12, -0.15, 0.5, -0.08, -0.16, -0.14, 1.1, 0.2, -0.1, -0.16, -0.13, -0.05, 0.75, -0.11,
    -0.16, -0.15, -0.09, 0.3, -0.13, -0.16, -0.07, 1.3, 0.35, -0.12, -0.15, -0.16, -0.1, 0.6, -0.04,
    -0.14, -0.16, -0.12, 0.15, -0.15, -0.11, 0.9, -0.09, -0.16, -0.13,
]  # fmt: skip


def _part(rows: list[dict], part: str, model: str = GPT2) -> list[dict]:
    """Return the rows of one part of the output for one model."""
    return [row for row in rows if row.get("part") == part and row.get("model") == model]


def _rest_rows(rows: list[dict]) -> tuple[list[dict], dict[int, list[dict]]]:
    """Return the trained rows, and the rows of the random seeds by layer."""
    rest = _part(rows, "rest")
    trained = sorted((r for r in rest if r["weights"] == "trained"), key=lambda r: r["layer"])
    fresh = {r["layer"]: [s for s in rest if s["weights"] != "trained" and s["layer"] == r["layer"]]
             for r in trained}  # fmt: skip
    return trained, fresh


def _gelu_sketch(fig: Figure, top: float, trained: dict, fresh: list[dict], dip: float) -> None:
    """Draw GELU over a sketch of one typical neuron's h, before and after training."""
    span = (-2.6, 1.3)
    upper = Panel(fig, (44, top, 300, 128), span, (-0.3, 1.2))
    lower = Panel(fig, (44, top + 146, 300, 90), span, (0, 1.1))
    upper.head("L9: where a typical neuron's h sits on GELU")
    upper.grid_y([0, 1], 0)
    upper.horizontal(0, "axis")
    xs = sample(*span)
    upper.curve(xs, [gelu(x) for x in xs], "curve")
    fresh_h, fresh_sd = mean(r["median_h"] for r in fresh), mean(r["median_h_sd"] for r in fresh)
    shapes = [(fresh_h, fresh_sd, "fneutral", "line sneutral"),
              (trained["median_h"], trained["median_h_sd"], "fc1", "line sc1")]  # fmt: skip
    for centre, sd, fill, line in shapes:
        ys = [normal(x, centre, sd) for x in xs]
        points = [(lower.x(x), lower.y(y)) for x, y in zip(xs, ys, strict=True)]
        base = [(lower.x(span[1]), lower.y(0)), (lower.x(span[0]), lower.y(0))]
        fig.path(points + base, f"{fill} soft", close=True)
        lower.curve(xs, ys, line)
        fig.line(lower.x(centre), upper.y(gelu(centre)), lower.x(centre), lower.y(0), line)
        fig.dot(upper.x(centre), upper.y(gelu(centre)), 5, f"{fill} ring")
    for panel in (upper, lower):
        panel.vertical(dip, "dip")
    fig.text(upper.x(dip) + 5, upper.top + 11, f"dip {number(dip)}", "small")
    lower.horizontal(0, "axis")
    lower.grid_x([-2, -1, 0, 1], 0)
    lower.x_title("pre-activation h (sketch: normal curves with L9's median and SD)")
    fig.text(lower.x(trained["median_h"]) - 48, lower.top + 22, "trained", "small", "end")
    fig.text(lower.x(fresh_h) + 34, lower.top + 22, "random init", "small")


def _rest_by_layer(
    fig: Figure, top: float, trained: list[dict], fresh: dict[int, list[dict]], dip: float
) -> None:
    """Draw the median h of each layer, trained and at random init."""
    panel = Panel(fig, (424, top, 312, 236), (-0.4, 11.4), (-1.45, 0.2))
    panel.head("Median h by layer")
    panel.grid_y([-1, -0.5, 0], 1)
    panel.grid_x([0, 3, 6, 9, 11], 0)
    panel.horizontal(dip, "dip")
    fig.text(panel.x(6), panel.y(dip) - 6, "dip", "small", "middle")
    layers = [r["layer"] for r in trained]
    init = [mean(s["median_h"] for s in fresh[i]) for i in layers]
    panel.curve(layers, init, "line sneutral")
    panel.curve(layers, [r["median_h"] for r in trained], "line sc1")
    for row in trained:
        fig.dot(panel.x(row["layer"]), panel.y(row["median_h"]), 4, "fc1 ring")
    fig.text(
        panel.x(1) + 8,
        panel.y(trained[1]["median_h"]) + 2,
        f"L1 {number(trained[1]['median_h'])}",
        "small",
    )
    fig.text(
        panel.x(9) - 4,
        panel.y(trained[9]["median_h"]) + 18,
        f"L9 {number(trained[9]['median_h'])}",
        "small",
        "middle",
    )
    fig.text(
        panel.left + panel.width, panel.y(init[-1]) - 8, "random init (5 seeds)", "small", "end"
    )
    panel.x_title("layer")


def rest(rows: list[dict], dip: float) -> Figure:
    """Draw where GPT-2's neurons rest, before and after training."""
    trained, fresh = _rest_rows(rows)
    fig = Figure(
        ["Training moves GPT-2's neurons left of GELU's dip"],
        [
            "At random init the median neuron rests at 0, on GELU's slope. Trained, it rests "
            "near −1 on most layers."
        ],
        "Left: GELU with a sketch of a typical L9 neuron's h, centred at 0 at random init and at "
        "minus 1.04 after training. Right: trained median h by layer is near minus 1 except at L1.",
    )
    top = fig.body_top + 28
    _gelu_sketch(fig, top, trained[9], fresh[9], dip)
    _rest_by_layer(fig, top, trained, fresh, dip)
    return fig


def _conditions() -> dict[str, list[float]]:
    """Return the toy activation under each clamp condition."""
    clean = ILLUSTRATION
    below = [a for a in clean if a < 0]
    level = mean(below)
    clamp = [level if a < 0 else a for a in clean]
    size = len(clean)
    change = [new - old for new, old in zip(clamp, clean, strict=True)]
    shuffled = [clean[i] + change[(i * 17 + 5) % size] for i in range(size)]
    return {
        "clean": clean,
        "clamp": clamp,
        "zero": [0.0 if a < 0 else a for a in clean],
        "mean": [mean(clean)] * size,
        "null": shuffled,
    }


def clamps() -> Figure:
    """Draw what each clamp condition does to one neuron over a run of tokens."""
    fig = Figure(
        ["What each intervention does to a neuron"],
        [
            "An illustration with made-up values, not data. Each bar is one token. Grey "
            "shows the clean activation."
        ],
        "Illustration of a neuron's activation over forty tokens under the clean run, the clamp, "
        "the zero, the mean and the shuffled null. The clamp removes variation and keeps the "
        "level. The zero removes the level too.",
    )
    fig.legend(
        fig.body_top + 10, [("fc1", "Negative side (a < 0)"), ("fc2", "Positive side (a > 0)")]
    )
    rows = [
        ("Clean", "clean", "The model as it is."),
        ("Clamp", "clamp", "Keeps the side's level, and removes its variation."),
        ("Zero", "zero", "Removes the level as well."),
        ("Mean", "mean", "Every token holds the neuron's mean."),
        ("Shuffled null", "null", "The clamp's changes, moved to other tokens."),
    ]
    values = _conditions()
    top = fig.body_top + 26
    for index, (name, key, note) in enumerate(rows):
        panel = Panel(fig, (40, top + index * 76, 400, 66), (0, 40), (-0.4, 1.4))
        fig.text(462, panel.y(0.5), name, "head")
        fig.text(462, panel.y(0.5) + 17, note, "small")
        panel.horizontal(0, "axis")
        for token, (old, new) in enumerate(zip(ILLUSTRATION, values[key], strict=True)):
            x = panel.x(token) + 1.5
            if key != "clean":
                fig.bar(x, panel.y(0), 7, panel.y(old) - panel.y(0), "fwash")
            fill = "fc2" if new > 0 else "fc1"
            fig.bar(x + (0 if key == "clean" else 1.5), panel.y(0), 7 if key == "clean" else 4,
                    panel.y(new) - panel.y(0), fill)  # fmt: skip
    return fig


def _cost_panel(fig: Figure, panel: Panel, rows: list[dict], name: str, labels: bool) -> None:
    """Draw the nats that each condition adds, for one model."""
    costs = {r["condition"]: r["added_ce"] for r in rows}
    panel.head(name)
    panel.grid_x([0, 2, 4, 6, 8], 0)
    panel.vertical(0, "axis")
    panel.x_title("added cross-entropy (nats)")
    gap = panel.height / len(COST_ROWS)
    for index, (label, condition, fill) in enumerate(COST_ROWS):
        y = panel.top + index * gap + 6
        if labels:
            fig.text(panel.left - 10, y + gap / 2 - 2, label, "", "end")
        width = panel.x(costs[condition]) - panel.left
        fig.bar(panel.left, y, width, gap - 12, fill)
        fig.text(panel.left + width + 5, y + gap / 2 - 2, number(costs[condition]), "small")


def cost(rows: list[dict]) -> Figure:
    """Draw the cost of each condition with all layers changed, for both models."""
    fig = Figure(
        ["Most of the negative side's cost in GPT-2 is its level, not its variation"],
        [
            "All layers changed at once. In GPT-2 a clamp costs 0.19 nats and a zero 3.72. "
            "In Qwen the two clamps cost the same."
        ],
        "Bar charts of added cross-entropy. In GPT-2 the negative clamp adds 0.19 nats, the "
        "negative zero 3.72, the positive clamp 1.34. In Qwen both clamps add 2.46.",
    )
    fig.legend(
        fig.body_top + 10,
        [
            ("fc1", "Negative side"),
            ("fc2", "Positive side"),
            ("fneutral", "Shuffled null"),
            ("fc3", "Both sides"),
        ],
    )
    top = fig.body_top + 44
    panels = columns(fig, 2, 176, top, 222, (0, 10), (0, 1), gap=34)
    _cost_panel(fig, panels[0], _part(rows, "all"), "GPT-2 small (GELU)", True)
    _cost_panel(fig, panels[1], _part(rows, "all", QWEN), "Qwen2.5-0.5B (SwiGLU)", False)
    return fig


def _ratio_panel(
    fig: Figure, panel: Panel, xs: list[float], ys: list[float], name: str, cls: str
) -> None:
    """Draw one KL ratio against the variance ratio, with one dot for each layer."""
    panel.head(name)
    panel.grid_x([0, 0.1, 0.2, 0.3], 1)
    panel.grid_y([0, 0.1, 0.2, 0.3], 1)
    panel.curve([0, 0.34], [0, 0.34], "axis")
    for x, y in zip(xs, ys, strict=True):
        fig.dot(panel.x(x), panel.y(y), 5, f"{cls} ring")
    fig.text(panel.x(xs[0]) + 9, panel.y(ys[0]) + 4, "L0", "small")
    fig.text(panel.left + 8, panel.top + 16, f"r = {number(correlation(xs, ys))}", "ink")
    fig.text(panel.x(0.27), panel.y(0.31), "x = y", "small", "end")
    panel.x_title("activation variance ratio, − / +")


def kl_ratio(rows: list[dict]) -> Figure:
    """Draw the KL ratio and the null's KL ratio against the variance ratio."""
    layers = _part(rows, "layer")
    xs = [r["act_variance_ratio"] for r in layers]
    fig = Figure(
        ["A shuffled change with no link to the token fits the variance just as well"],
        [
            'So "the same signal, scaled down" gets no support. Any change of this size gives '
            "this pattern. Each dot is a layer.",
        ],
        "Two scatter plots against the activation variance ratio for 12 layers. The clamp's KL "
        "ratio has r 0.95 and the shuffled null's has r 0.97. Both lie near the diagonal.",
    )
    panels = columns(fig, 2, 48, fig.body_top + 26, 262, (0, 0.34), (0, 0.34), gap=36)
    _ratio_panel(
        fig, panels[0], xs, [r["kl_ratio"] for r in layers], "KL ratio of the clamp", "fc1"
    )
    null = [r["negative_null_kl"] / r["positive_null_kl"] for r in layers]
    _ratio_panel(fig, panels[1], xs, null, "KL ratio of the shuffled null", "fneutral")
    return fig
