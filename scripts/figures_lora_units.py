"""The figures of papers/lora-units/README.md, from its results (#5166).

An adapter direction is a triangle in the second colour, and a native pole a circle in the
first, as in papers/rome-keep (figures_rome_keep). A unit whose explanation passed the review is
filled in its arm's colour. One that passed some tests is grey, and one that failed is pale.
"""

import math

from figure_motion import reveal
from figure_svg import Figure, Panel
from figures_random_change_keep import _mark

CYCLE = "12s"
ARMS = {"adapter": ("random", "c2"), "native": ("instruct", "c1")}
NAMES = {"adapter": "rank-1 LoRA down_proj directions", "native": "random native poles"}
SHORT = {"adapter": "rank-1 LoRA directions", "native": "random native poles"}
FILLS = {"NEEDS_REVISION": "fneutral", "REJECTED": "fwash"}


def _units(rows: list[dict], arm: str) -> list[dict]:
    """One arm's ``unit`` rows, by layer.

    Args:
        rows: The paper's results.
        arm: ``adapter`` or ``native``.
    """
    return sorted((r for r in rows if r["table"] == "unit" and r["arm"] == arm),
                  key=lambda r: (r["layer"], r["unit"]))  # fmt: skip


def _arm(rows: list[dict], arm: str) -> dict:
    """One arm's ``arm`` row.

    Args:
        rows: The paper's results.
        arm: ``adapter`` or ``native``.
    """
    return next(r for r in rows if r["table"] == "arm" and r["arm"] == arm)


def _legend(fig: Figure, y: float) -> None:
    """Each arm's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for arm, (which, colour) in ARMS.items():
        _mark(fig, which, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, NAMES[arm], "small")
        x += 30 + 5.8 * len(NAMES[arm])
    fig.bottom = y


def verdicts(rows: list[dict]) -> Figure:
    """Each audited unit as one square, coloured by its verdict, arm by arm.

    Args:
        rows: The paper's results.

    Returns:
        The figure. The squares appear one by one, in layer order, arm by arm.
    """
    arms = {arm: _arm(rows, arm) for arm in ARMS}
    a, n = arms["adapter"], arms["native"]
    fig = Figure(
        ["The same audit verifies far fewer LoRA directions than native neurons"],
        ["Each square is one unit, by layer. Filled: passed the review. Grey: passed some "
         "tests. Pale: failed."],
        f"Two rows of squares. Of {a['units']} adapter directions, {a['verified']} passed the "
        f"review, {a['needs_revision']} passed some tests and {a['rejected']} failed. Of "
        f"{n['units']} native poles, {n['verified']} passed, {n['needs_revision']} passed some "
        f"tests and {n['rejected']} failed.",
    )  # fmt: skip
    size, gap, left = 21, 4, 225
    for i, arm in enumerate(ARMS):
        y = fig.body_top + 30 + i * 70
        units = _units(rows, arm)
        fig.text(20, y + 16, SHORT[arm], "head")
        fig.text(20, y + 32, f"{arms[arm]['verified']} of {len(units)} passed", "small")
        for j, unit in enumerate(units):
            shown = f"u{i}x{j}"
            fig.motion += reveal(shown, 4 + 44 * i + 40 * j / len(units), CYCLE)
            fill = f"f{ARMS[arm][1]}" if unit["status"] == "VERIFIED" else FILLS[unit["status"]]
            fig.bar(left + j * (size + gap), y, size, size, f"{fill} {shown}")
            fig.text(left + j * (size + gap) + size / 2, y + size + 12, str(unit["layer"]),
                     f"small {shown}", "middle")  # fmt: skip
    fig.text(left, fig.body_top + 30 + 2 * 70 + 6, "number under each square: its layer",
             "small")  # fmt: skip
    fig.bottom = fig.body_top + 30 + 2 * 70 + 6
    return fig


def gates(rows: list[dict]) -> Figure:
    """Each unit's specificity against its sensitivity, with the review's two lines.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    a, n = _arm(rows, "adapter"), _arm(rows, "native")
    fig = Figure(
        ["The LoRA directions fail on the hard negatives, not on the held-out positives"],
        ["Each mark is one unit: the share of hard negatives it stays inactive on, and the "
         "share of",
         "held-out positives it fires on. A unit must reach both grey lines to pass."],
        f"A scatter of {a['units'] + n['units']} units. The adapter directions sit at a mean "
        f"specificity of {a['mean_specificity']:.2f} and sensitivity of "
        f"{a['mean_sensitivity']:.2f}. The native poles sit at {n['mean_specificity']:.2f} "
        f"and {n['mean_sensitivity']:.2f}.",
    )  # fmt: skip
    panel = Panel(fig, (80, fig.body_top + 30, 560, 300), (-0.03, 1.03), (-0.03, 1.03))
    panel.grid_x([0, 0.2, 0.4, 0.6, 0.8, 1.0], 1)
    panel.grid_y([0, 0.2, 0.4, 0.6, 0.8, 1.0], 1)
    panel.x_title("specificity: hard negatives that stay inactive")
    fig.text(panel.x(-0.03) - 50, panel.y(1.03) - 10, "sensitivity", "small")
    panel.vertical(0.8, "dip")
    panel.horizontal(0.6, "dip")
    for arm, (which, colour) in ARMS.items():
        for k, unit in enumerate(_units(rows, arm)):
            dx = ((k * 7) % 5 - 2) * 0.004  # a small, fixed spread, so equal units both show
            _mark(fig, which, panel.x(unit["specificity"] + dx), panel.y(unit["sensitivity"]),
                  4.5, f"f{colour} ring")  # fmt: skip
    fig.text(panel.x(0.81), panel.y(1.03) - 10, "passes both ↓", "small")
    _legend(fig, panel.top + panel.height + 52)
    return fig


def _log(value: float) -> float:
    """The base-10 logarithm, with a floor at one token in 100,000.

    Args:
        value: A share.

    Examples:
        >>> _log(0.01), _log(0.0)
        (-2.0, -5.0)
    """
    return math.log10(max(value, 1e-5))


def density(rows: list[dict]) -> Figure:
    """Each unit's active share of its scan's tokens, on a log scale, arm by arm.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    a, n = _arm(rows, "adapter"), _arm(rows, "native")
    fig = Figure(
        [f"The LoRA directions are active on about "
         f"{a['mean_active_share'] / n['mean_active_share']:.0f} times as many tokens"],
        ["Each mark is one unit: the share of its scan's tokens at or above a quarter of its "
         "maximum,",
         "the review's line for active. The bars are each arm's mean."],
        f"A strip chart on a log scale. The adapter directions are active on a mean of "
        f"{a['mean_active_share']:.1%} of tokens, and the native poles on "
        f"{n['mean_active_share']:.2%}.",
    )  # fmt: skip
    panel = Panel(fig, (230, fig.body_top + 30, 480, 120), (-4.2, 0), (0, 2))
    for tick, label in ((-4, "0.01%"), (-3, "0.1%"), (-2, "1%"), (-1, "10%"), (0, "100%")):
        panel.vertical(tick, "grid")
        fig.text(panel.x(tick), panel.top + panel.height + 15, label, "small", "middle")
    panel.x_title("active share of the scan's tokens (log scale)")
    for i, (arm, (which, colour)) in enumerate(ARMS.items()):
        y = panel.y(1.5 - i)
        fig.text(20, y + 4, NAMES[arm], "head")
        for k, unit in enumerate(_units(rows, arm)):
            jitter = ((k * 7) % 5 - 2) * 4
            _mark(fig, which, panel.x(_log(unit["active_share"])), y + jitter, 4.5,
                  f"f{colour} ring")  # fmt: skip
        x = panel.x(_log(_arm(rows, arm)["mean_active_share"]))
        fig.line(x, y - 18, x, y + 18, f"range s{colour}")
    fig.bottom = panel.top + panel.height + 20
    return fig


def loading(loadings: list[dict]) -> Figure:
    """Each layer's median percentile of its VERIFIED neurons in the adapter's ``|lora_A|``.

    Args:
        loadings: The ``loading`` rows of results.units.jsonl.

    Returns:
        The figure.
    """
    rows = sorted((r for r in loadings if r["table"] == "loading"
                   and r["median_percentile"] is not None), key=lambda r: r["layer"])  # fmt: skip
    middle = sorted(r["median_percentile"] for r in rows)[len(rows) // 2]
    fig = Figure(
        ["The adapters put no extra weight on the explained neurons"],
        ["For each layer: the median rank, from 0 to 1, of the layer's explained neurons in "
         "|lora_A|.",
         "Random neurons give 0.5. The mark's size grows with the number of explained neurons."],
        f"A dot per layer, for {len(rows)} layers. The medians scatter around 0.5, and the "
        f"median of the {len(rows)} layer medians is {middle:.2f}.",
    )  # fmt: skip
    panel = Panel(fig, (80, fig.body_top + 30, 600, 220), (-0.8, 23.8), (0, 1))
    panel.grid_x([0, 4, 8, 12, 16, 20, 23], 0)
    panel.grid_y([0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.x_title("layer")
    panel.horizontal(0.5, "dip")
    for r in rows:
        fig.dot(panel.x(r["layer"]), panel.y(r["median_percentile"]),
                2.5 + 1.2 * math.sqrt(r["verified"]), "fc2 ring")  # fmt: skip
    fig.bottom = panel.top + panel.height + 20
    return fig


def build(rows: list[dict], loadings: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.
        loadings: The rows of results.units.jsonl.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"verdicts.svg": verdicts(rows), "gates.svg": gates(rows),
            "density.svg": density(rows), "loading.svg": loading(loadings)}  # fmt: skip
