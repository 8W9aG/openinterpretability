"""The dormant-pole and effective-rank figure of papers/neuron-development (#5134)."""

import math

from figure_svg import Figure, Panel
from figures_development import TICKS, _label, _x


def _ceiling(value: float, step: float) -> float:
    """``value`` rounded up to a multiple of ``step``, at least ``step``.

    Args:
        value: The data's maximum.
        step: The multiple.

    Returns:
        The axis's top.
    """
    return max(step, math.ceil(value / step) * step)


def _axes(panel: Panel, head: str, ticks: tuple) -> None:
    """The panel's head, its log-step grid and labels, and its y grid.

    Args:
        panel: The panel.
        head: Its name.
        ticks: The y grid's values.
    """
    panel.head(head)
    for tick in TICKS:
        panel.figure.line(panel.x(_x(tick)), panel.top, panel.x(_x(tick)),
                          panel.top + panel.height, "grid")  # fmt: skip
        panel.figure.text(panel.x(_x(tick)), panel.top + panel.height + 15, _label(tick),
                          "small", "middle")  # fmt: skip
    panel.x_title("training step (log scale)")
    panel.grid_y(ticks, places=0)


def _layers(panel: Panel, rows: list[dict], key: str, scale: float) -> None:
    """One line for each layer, labelled at its right end, so no line rests on colour.

    Args:
        panel: The panel.
        rows: The ``layer`` rows.
        key: The measure to draw.
        scale: The factor from the measure to the axis.
    """
    ends = []
    for layer in sorted({r["layer"] for r in rows}):
        own = sorted((r for r in rows if r["layer"] == layer), key=lambda r: r["step"])
        panel.curve([_x(r["step"]) for r in own], [r[key] * scale for r in own], "curve")
        ends.append((panel.y(own[-1][key] * scale), f"L{layer}"))
    last = -math.inf
    for y, name in sorted(ends):  # top to bottom, 11 px apart so no labels overlap
        y = max(y + 4, last + 11)
        panel.figure.text(panel.x(panel.x_range[1]) + 6, y, name, "small")
        last = y


def _peaks(panel: Panel, rows: list[dict], key: str, scale: float, floor: float) -> None:
    """One line for each layer, labelled with its peak by a leader line.

    The layers whose peak stays under ``floor`` share one note.

    Args:
        panel: The panel.
        rows: The ``layer`` rows.
        key: The measure to draw.
        scale: The factor from the measure to the axis.
        floor: The peak, on the axis, under which a layer gets no label of its own.
    """
    flat = []
    for layer in sorted({r["layer"] for r in rows}):
        own = sorted((r for r in rows if r["layer"] == layer), key=lambda r: r["step"])
        panel.curve([_x(r["step"]) for r in own], [r[key] * scale for r in own], "curve")
        top = max(own, key=lambda r: r[key])
        if top[key] * scale < floor:
            flat.append(f"L{layer}")
            continue
        x, y = panel.x(_x(top["step"])), panel.y(top[key] * scale)
        panel.figure.line(x + 3, y, panel.x(_x(400)) - 4, y, "axis")  # a leader to the label
        panel.figure.text(panel.x(_x(400)), y + 4, f"L{layer}, {top[key] * scale:.1f}%", "small")
    if flat:
        panel.figure.text(panel.x(_x(400)), panel.y(panel.y_range[1] * 0.93),
                          f"{', '.join(flat)}: under {floor:g}% throughout", "small")  # fmt: skip


def figure(rows: list[dict]) -> Figure:
    """Each layer's dormant share and effective rank, by training step.

    Args:
        rows: The ``layer`` rows of ``results.dormant.jsonl``.

    Returns:
        The figure.
    """
    rows = [r for r in rows if r.get("table") == "layer"]
    # Round each axis up from the data, so a new run that peaks higher still fits.
    top_share = _ceiling(max(r["dormant"] for r in rows) * 100, 5)
    top_rank = _ceiling(max(r["erank"] for r in rows), 200)
    fig = Figure(
        ["Pythia-70m's deep MLP layers collapse early, recover, then slowly lose rank"],
        [
            "Left: the share of each layer's 4,096 poles that are active (by the sign of the",
            "activation) on under 1% of 18,782 tokens. Right: the effective rank of each",
            "layer's activations, on a seeded sample of 2,000 tokens. One line for each",
            "layer, labelled at its peak (left) or at its right end (right).",
        ],
        "Two panels of six lines each, one for each layer of Pythia-70m, over 20 training "
        "checkpoints. Dormant poles rise to 11.6% in layer 5 at step 32 and are gone by step "
        "512. Effective rank dips at steps 32 to 64, peaks near step 4,000, then falls slowly "
        "in layers 2 to 5.",
    )
    # The same log-step axis as the paper's other figures; ranks start at 0, not at the data.
    span = (0.0, _x(150_000))
    left = Panel(fig, (64, fig.body_top + 28, 290, 260), span, (0.0, top_share))
    right = Panel(fig, (440, fig.body_top + 28, 270, 260), span, (0.0, top_rank))
    _axes(left, "Dormant poles (%)", tuple(range(0, int(top_share) + 1, int(top_share) // 5)))
    _axes(right, "Effective rank", tuple(range(0, int(top_rank) + 1, int(top_rank) // 4)))
    _peaks(left, rows, "dormant", 100, 1)
    _layers(right, rows, "erank", 1)
    fig.bottom = left.top + left.height + 36
    return fig
