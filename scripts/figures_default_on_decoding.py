"""The figures of papers/default-on-poles/README.md that compare a real side with a null."""

from figure_svg import Figure, Panel, number
from figures_default_on import CONCEPT_LAYERS, _part

NULLS = ("null_0", "null_1", "null_2")


def tails(rows: list[dict]) -> Figure:
    """Draw the reliability of real neurons and of random directions at L9."""
    population = _part(rows, "band_population")[0]
    neurons = _part(rows, "band_neuron")
    studied = next(r for r in neurons if r["neuron"] == 1046)
    controls = [r["reliability"] for r in neurons if r["neuron"] != 1046]
    fig = Figure(
        ["A random direction's low tail is as reliable by token as a real neuron's"],
        [
            "Split-half reliability by token at L9. A bar is the 5th–95th percentile, and "
            "its dot is the median."
        ],
        "Interval plot of reliability. All 2,972 default-on neurons have median 0.77, random "
        "directions median 0.77. N1046 is 0.70, and random directions at its share have "
        "median 0.68.",
    )
    top = fig.body_top + 10
    panel = Panel(fig, (246, top, 490, 180), (0.45, 0.95), (0, 1))
    panel.grid_x([0.5, 0.6, 0.7, 0.8, 0.9], 1)
    panel.x_title("split-half reliability by token")
    entries = [
        ("All 2,972 default-on neurons", population["reliability_quantiles"], "c1"),
        ("Random directions, same shares", population["random_direction_quantiles"], "neutral"),
        ("Random directions at N1046's "
"share", population["random_direction_at_studied_share"], "neutral"),
    ]  # fmt: skip
    for index, (label, quantiles, colour) in enumerate(entries):
        y = top + (index + 0.5) * 36
        fig.text(236, y + 4, label, "", "end")
        fig.line(panel.x(quantiles["q05"]), y, panel.x(quantiles["q95"]), y, f"range s{colour} ")
        fig.dot(panel.x(quantiles["q50"]), y, 5, f"f{colour} ring")
        fig.text(panel.x(quantiles["q95"]) + 8, y + 4, number(quantiles["q50"]), "small")
    y = top + 3.5 * 36
    fig.text(236, y + 4, "The 10 controls", "", "end")
    for value in controls:
        fig.dot(panel.x(value), y, 4, "fc1 ring")
    y = top + 4.5 * 36
    fig.text(236, y + 4, "N1046", "head", "end")
    fig.dot(panel.x(studied["reliability"]), y, 6, "fc1 ring")
    fig.text(panel.x(studied["reliability"]) + 10, y + 4, number(studied["reliability"]), "small")
    return fig


def _lead_panel(fig: Figure, panel: Panel, rows: list[dict]) -> None:
    """Draw each side's lead in R² over the best of three null draws, by layer."""
    layers = [r["layer"] for r in rows]
    best = [max(r[n]["log_prob_r2"] for n in NULLS) for r in rows]
    worst = [min(r[n]["log_prob_r2"] for n in NULLS) for r in rows]
    panel.head("Next-token distribution: R² lead over the null")
    panel.grid_y([0, 0.02, 0.04, 0.06], 2)
    panel.grid_x([0, 3, 6, 9, 11], 0)
    spread = [(panel.x(i), panel.y(w - b)) for i, w, b in zip(layers, worst, best, strict=True)]
    base = [(panel.x(layers[-1]), panel.y(0)), (panel.x(layers[0]), panel.y(0))]
    fig.path(spread + base, "fneutral soft", close=True)
    panel.horizontal(0, "axis")
    fig.text(
        panel.left + panel.width,
        panel.y(-0.027),
        "shaded: spread of the 3 null draws",
        "small",
        "end",
    )
    for side, line, fill in (("positive", "line sc2", "fc2"), ("negative", "line sc1", "fc1")):
        lead = [r[side]["log_prob_r2"] - b for r, b in zip(rows, best, strict=True)]
        panel.curve(layers, lead, line)
        for layer, value in zip(layers, lead, strict=True):
            fig.dot(panel.x(layer), panel.y(value), 3.5, f"{fill} ring")
        fig.text(panel.x(layers[-1]) + 8, panel.y(lead[-1]) + 4, number(lead[-1], 3), "small")
    panel.x_title("layer")


def _label_lead(row: dict) -> float:
    """Return the negative side's lead in mean ambiguous AUROC over the best null draw."""

    def score(entry: dict) -> float:
        values = [value for value in entry["auroc_ambiguous"] if value is not None]
        return sum(values) / len(values)

    return score(row["negative"]) - max(score(row[n]) for n in NULLS)


def _label_panel(fig: Figure, panel: Panel, rows: list[dict]) -> None:
    """Draw the negative side's lead on human labels, one dot for each dataset and layer."""
    panel.head("Human labels: AUROC lead, − side")
    panel.grid_y([0, 0.02, 0.04, 0.06], 2)
    panel.horizontal(0, "axis")
    for index, layer in enumerate(CONCEPT_LAYERS):
        panel.figure.text(
            panel.x(index), panel.top + panel.height + 15, f"L{layer}", "small", "middle"
        )
        for row in (r for r in rows if r["layer"] == layer):
            fig.dot(panel.x(index), panel.y(_label_lead(row)), 4, "fc1 ring")
    panel.x_title("each dot is one dataset")
    largest = max(_label_lead(row) for row in rows)
    fig.text(panel.x(4.4), panel.y(0.03), f"largest: {number(largest, 3)}", "small", "end")


def decoding(rows: list[dict]) -> Figure:
    """Draw how far each side leads random features, on the two new targets."""
    fig = Figure(
        ["The negative side leads random features only a little on these targets"],
        [
            "Zero is the best of three random-feature draws. Late layers show the positive "
            "side pulling away.",
            "The two panels use different scores, so compare each with its own zero, not "
            "with the other.",
        ],
        "Left: R² lead by layer. The negative side leads by at most 0.025, and the positive side "
        "reaches 0.073 at L11. Right: the negative side's AUROC lead on labels is at most 0.011.",
    )
    fig.legend(fig.body_top + 10, [("fc1", "Negative side"), ("fc2", "Positive side")])
    top = fig.body_top + 44
    left = Panel(fig, (56, top, 400, 196), (-0.4, 12.6), (-0.032, 0.08))
    _lead_panel(fig, left, _part(rows, "targets_next_token"))
    right = Panel(fig, (526, top, 210, 196), (-0.5, 4.5), (-0.032, 0.08))
    _label_panel(fig, right, _part(rows, "targets_concept"))
    return fig
