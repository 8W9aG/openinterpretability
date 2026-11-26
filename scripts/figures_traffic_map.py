"""The figures of papers/traffic-map/README.md, from its results.jsonl (#5231).

A neuron site of the even draw is a circle in the first colour. A head is a triangle in
the second colour, and a top-traffic neuron a square in the third. The shape keeps them
apart without colour.
"""

import math

from figure_motion import counter, reveal
from figure_svg import Figure, Panel
from traffic_stats import PREDICTORS, correlation, tested_sites, within_layer

CYCLE = "9s"
DRAWS = (("decile", "c1", "neurons, even draw over the deciles of traffic"),
         ("head", "c2", "attention heads, all 48"),
         ("top", "c3", "the 10 neurons of most traffic"))  # fmt: skip
NAMES = {"traffic": "traffic", "unweighted": "traffic, no use weight",
         "count": "poles reached (depth)", "loss_fisher": "diagonal Fisher of the loss",
         "inverse_weight_norm": "change / weight norm"}  # fmt: skip
FLOOR = -2.5  # log10 of the traffic where a site of no traffic is drawn


def _shape(fig: Figure, draw: str, x: float, y: float, size: float, style: str) -> None:
    """One site's mark: a circle, a triangle or a square, by its draw.

    Args:
        fig: The figure.
        draw: ``decile``, ``head`` or ``top``.
        x: The mark's centre, in pixels.
        y: The mark's centre, in pixels.
        size: The mark's radius, in pixels.
        style: The mark's CSS classes.
    """
    if draw == "decile":
        fig.dot(x, y, size, style)
    elif draw == "head":
        h = size * 1.2
        fig.path([(x, y - h), (x + h, y + h * 0.8), (x - h, y + h * 0.8)], style, close=True)
    else:
        fig.path([(x - size, y - size), (x + size, y - size), (x + size, y + size),
                  (x - size, y + size)], style, close=True)  # fmt: skip


def _log(traffic: float) -> float:
    """log10 of the traffic, with no traffic at :data:`FLOOR`.

    Args:
        traffic: A site's traffic.
    """
    return math.log10(traffic) if traffic > 0 else FLOOR


def traffic(rows: list[dict]) -> Figure:
    """Each tested site's drift against its traffic, one draw at a time.

    Args:
        rows: The paper's results.

    Returns:
        The figure. It shows the even draw, then the heads, then the top sites, and rests
        with all three.
    """
    sites = tested_sites(rows)
    top = [s for s in sites if s["draw"] == "top"]
    fig = Figure(
        ["The more traffic a site carries, the further one random change",
         "there moves the registry's explanations"],
        ["Each mark is one site of pythia-70m, changed by Gaussian noise of norm 1. Up: the",
         "mean drift of the 176 VERIFIED poles' probe peaks, as a share of each pole's scale."],
        f"A scatter of {len(sites)} sites, traffic on a log scale against mean drift. Drift "
        f"rises with traffic. The {len(top)} top-traffic neurons sit highest, up to "
        f"{max(s['mean_drift'] for s in top):.1%} of scale.",
    )  # fmt: skip
    high = max(s["mean_drift"] for s in sites)
    panel = Panel(fig, (80, fig.body_top + 30, 600, 260), (FLOOR - 0.3, 1.5),
                  (0, high * 1.08))  # fmt: skip
    panel.grid_y([0, 0.01, 0.02, 0.03], 2)
    for tick in (-2, -1, 0, 1):
        panel.vertical(tick, "grid")
        fig.text(panel.x(tick), panel.top + panel.height + 15, f"{10.0**tick:g}", "small",
                 "middle")  # fmt: skip
    fig.text(panel.x(FLOOR), panel.top + panel.height + 15, "none", "small", "middle")
    panel.x_title("traffic, Σ use × ‖∂R/∂θ‖ (log scale)")
    starts = [8.0, 36.0, 64.0]
    labels = counter("step", starts, CYCLE)
    for i, ((draw, colour, label), (cls, css)) in enumerate(zip(DRAWS, labels, strict=True)):
        shown = f"draw{i}"
        fig.motion += reveal(shown, starts[i], CYCLE) + css
        part = [s for s in sites if s["draw"] == draw]
        fig.text(panel.left, panel.top - 10, f"now: {label} ({len(part)})", f"small {cls}")
        for s in part:
            _shape(fig, draw, panel.x(_log(s["traffic"])), panel.y(s["mean_drift"]), 4.0,
                   f"f{colour} ring {shown}")  # fmt: skip
    best = max(top, key=lambda s: s["traffic"])
    last = f"small draw{len(DRAWS) - 1}"  # the label shows with the top sites
    fig.text(panel.x(_log(best["traffic"])) - 8, panel.y(best["mean_drift"]) + 4,
             f"L{best['layer']}/N{best['index']}", last, "end")  # fmt: skip
    _legend(fig, panel.top + panel.height + 52)
    return fig


def _legend(fig: Figure, y: float) -> None:
    """Each draw's shape and name, in one row, at the figure's foot.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for draw, colour, label in DRAWS:
        _shape(fig, draw, x + 5, y - 4, 4.5, f"f{colour}")
        fig.text(x + 15, y, label, "small")
        x += 30 + 5.6 * len(label)
    fig.bottom = y


def predictors(rows: list[dict]) -> Figure:
    """Each predictor's ρ with drift and with flips, over all sites and within layers.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    sites = [s for s in tested_sites(rows) if s["draw"] in ("decile", "head")]
    measures = (("mean_drift", "mean drift"), ("flips", "flips"))
    fig = Figure(
        ["Much of what predicts the damage is depth;",
         "within a layer, traffic and the loss Fisher predict it about as well"],
        [f"Spearman's ρ with each measure over the {len(sites)} sites of the even draw. A dot "
         "is ρ over all sites, with",
         "a 95% bootstrap interval. A ring is ρ within each kind and layer, which gives "
         "depth no credit."],
        "Two panels, drift and flips. Over all sites, traffic, its unweighted form and the "
        "count of poles reached reach ρ 0.66 to 0.93. Within layers, traffic and the loss "
        "Fisher reach 0.25 to 0.55, and the change's relative size -0.17 to -0.04.",
    )  # fmt: skip
    room, row = 200.0, 26.0
    top = fig.body_top + 40
    for m, (measure, title) in enumerate(measures):
        left = room + m * 280
        panel = Panel(fig, (left, top, 220, row * len(PREDICTORS)), (-0.5, 1.0),
                      (len(PREDICTORS), 0))  # fmt: skip
        panel.grid_x([-0.5, 0, 0.5, 1.0], 1)
        panel.head(title)
        panel.vertical(0, "dip")
        _rows(fig, panel, sites, measure)
        if m == 0:
            for i, predictor in enumerate(PREDICTORS):
                fig.text(room - 12, panel.y(i + 0.5) + 4, NAMES[predictor], "", "end")
    bottom = top + row * len(PREDICTORS) + 50
    fig.legend(bottom, [("fc1", "over all sites, with 95% interval"),
                        ("fc2", "within each kind and layer")])  # fmt: skip
    fig.text(20, bottom + 22, "flat: the count is the same for every site of a layer, so it "
             "has no rank within one.", "small")  # fmt: skip
    fig.bottom = bottom + 22
    return fig


def _rows(fig: Figure, panel: Panel, sites: list[dict], measure: str) -> None:
    """One row for each predictor: ρ over all sites with its interval, and ρ within layers.

    Args:
        fig: The figure.
        panel: The measure's panel.
        sites: The sites of the even draw.
        measure: The measure of the panel.
    """
    for i, predictor in enumerate(PREDICTORS):
        y = panel.y(i + 0.5)
        raw = correlation(sites, predictor, measure)
        low, high = raw["ci95"]
        fig.line(panel.x(low), y, panel.x(high), y, "line sc1")
        fig.dot(panel.x(raw["rho"]), y, 4.5, "fc1 ring")
        local = within_layer(sites, predictor, measure)["rho"]
        if local is None:
            fig.text(panel.x(1.0) + 4, y + 4, "flat", "small")
        else:
            fig.dot(panel.x(local), y, 4.5, "fc2 ring")


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"traffic.svg": traffic(rows), "predictors.svg": predictors(rows)}
