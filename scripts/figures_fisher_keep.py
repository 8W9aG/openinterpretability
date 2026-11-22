"""The figures of papers/fisher-keep/README.md, from its results.jsonl."""

from figure_motion import counter
from figure_svg import Figure, Panel, number

# One colour and one shape per outcome through the paper. The shape keeps them apart
# without colour: circles keep, triangles fail.
OUTCOMES = ((True, "c1", "keeps"), (False, "c2", "fails"))
NULL_RANGE = (0.40, 0.65)  # the AUC axis of the null figure
NULL_BINS = 50
NULL_STEPS = (30, 100, 300, 1000, 3000, 10_000)  # shuffles shown, one frame each
CYCLE = "9s"
HALF = ".half{opacity:.55}"  # dense marks, see-through so overlaps show


def _mark(fig: Figure, keeps: bool, x: float, y: float, size: float, style: str) -> None:
    """One pole's mark: a circle when it keeps, a triangle when it fails.

    Args:
        fig: The figure.
        keeps: Whether the pole keeps its explanation.
        x: The mark's centre, in pixels.
        y: The mark's centre, in pixels.
        size: The mark's radius, in pixels.
        style: The mark's CSS classes.
    """
    if keeps:
        fig.dot(x, y, size, style)
    else:
        h = size * 1.15
        fig.path([(x, y - h), (x + h, y + h * 0.8), (x - h, y + h * 0.8)], style, close=True)


def _legend(fig: Figure, y: float) -> None:
    """The legend: each outcome's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for keeps, colour, label in OUTCOMES:
        _mark(fig, keeps, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, label)
        x += 26 + 6.6 * len(label)


def _test(rows: list[dict], outcome: str, statistic: str, fisher: str, weights: str) -> dict:
    """The ``test`` row of one outcome, statistic, Fisher and set of weights.

    Args:
        rows: The paper's results.
        outcome: The outcome.
        statistic: The statistic.
        fisher: The Fisher's kind of label, or ``rms``.
        weights: The weights.

    Returns:
        The row.
    """
    return next(r for r in rows if r["table"] == "test" and r["outcome"] == outcome
                and r["statistic"] == statistic and r["fisher"] == fisher
                and r["weights"] == weights)  # fmt: skip


def _median(values: list[float]) -> float:
    """The median.

    Args:
        values: The values, at least one.

    Returns:
        The median.
    """
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def keep(rows: list[dict]) -> Figure:
    """Each pole's within-layer Fisher percentile, for the poles that keep and that fail.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    poles = [r for r in rows if r["table"] == "pole"]
    t = _test(rows, "keeps", "auc", "model", "total")
    fig = Figure(
        ["A neuron's Fisher importance does not tell", "whether its explanation keeps"],
        [
            "Each mark is a VERIFIED pole of Qwen2.5-0.5B: its neuron's Fisher, as a percentile "
            "within its layer.",
            f"AUC {number(t['observed'], 3)} against a within-layer null of "
            f"{number(t['null_mean'], 3)} (p = {number(t['p'], 2)}). The ticks are medians.",
        ],
        f"A strip plot of the within-layer Fisher percentile of 311 poles, in two rows: the "
        f"{sum(p['keeps'] for p in poles)} that keep their explanation on the instruct model "
        f"and the {sum(not p['keeps'] for p in poles)} that fail. The two rows spread over the "
        f"same range, with similar medians. The AUC is {number(t['observed'], 3)}, and the null "
        f"mean is {number(t['null_mean'], 3)}.",
    )
    fig.motion += HALF
    panel = Panel(fig, (90, fig.body_top + 20, 640, 170), (0, 1), (0, 2))
    panel.grid_x([0, 0.25, 0.5, 0.75, 1], 2)
    panel.x_title("percentile of the neuron's Fisher within its layer")
    for slot, (keeps, colour, label) in enumerate(OUTCOMES):
        values = [p["percentile"]["model"]["total"] for p in poles if p["keeps"] is keeps]
        centre = 1.5 - slot
        fig.text(panel.left - 10, panel.y(centre) + 4, f"{label} ({len(values)})", "small", "end")
        for i, v in enumerate(sorted(values)):
            jitter = ((i * 37) % 11 / 10 - 0.5) * 0.6  # a fixed spread, not random
            _mark(fig, keeps, panel.x(v), panel.y(centre + jitter), 2.8, f"f{colour} half")
        mid = _median(values)
        fig.line(panel.x(mid), panel.y(centre + 0.42), panel.x(mid), panel.y(centre - 0.42),
                 f"line s{colour}")  # fmt: skip
    bottom = panel.top + panel.height + 52
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def _shares(values: list[float]) -> list[float]:
    """The share of ``values`` in each of :data:`NULL_BINS` bins over :data:`NULL_RANGE`.

    Args:
        values: The null values, at least one.

    Returns:
        One share for each bin. A value outside the range counts in the nearer end bin.
    """
    low, high = NULL_RANGE
    out = [0] * NULL_BINS
    for v in values:
        out[min(max(int((v - low) / (high - low) * NULL_BINS), 0), NULL_BINS - 1)] += 1
    return [c / len(values) for c in out]


def null(rows: list[dict]) -> Figure:
    """The within-layer null of the keep AUC, built up shuffle by shuffle, and the observed AUC.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    t = _test(rows, "keeps", "auc", "model", "total")
    values = t["null"]
    low, high = NULL_RANGE
    width = (high - low) / NULL_BINS
    frames = [_shares(values[:n]) for n in NULL_STEPS]
    top = max(max(f) for f in frames) * 1.1
    fig = Figure(
        ["The observed AUC sits inside its null"],
        [
            "Bars: the AUC after shuffling keep and fail within each layer, as a share of the "
            "shuffles so far.",
            f"Line: the observed AUC, {number(t['observed'], 3)}. "
            f"Two-sided p = {number(t['p'], 2)}.",
        ],
        f"A histogram of the null AUC that builds up over {len(values)} shuffles of the keep "
        f"labels within each layer. It centres near {number(t['null_mean'], 3)}. A vertical line "
        f"marks the observed AUC of {number(t['observed'], 3)}, inside the bulk of the null.",
    )
    panel = Panel(fig, (60, fig.body_top + 20, 670, 200), NULL_RANGE, (0, top))
    panel.grid_x([0.40, 0.45, 0.50, 0.55, 0.60, 0.65], 2)
    panel.x_title("AUC: P(the Fisher percentile of a pole that keeps > that of a pole that fails)")
    labels = counter("step", [100 * i / len(NULL_STEPS) for i in range(len(NULL_STEPS))], CYCLE)
    for (cls, css), frame, n in zip(labels, frames, NULL_STEPS, strict=True):
        fig.motion += css
        for b, share in enumerate(frame):
            if share:
                x = panel.x(low + b * width)
                fig.bar(x + 0.5, panel.y(0), panel.x(low + width) - panel.x(low) - 1,
                        panel.y(share) - panel.y(0), f"fneutral {cls}")  # fmt: skip
        fig.text(panel.left + panel.width, panel.top + 4, f"{n:,} shuffles", f"small {cls}", "end")
    panel.vertical(t["observed"], "line sc2")
    fig.text(panel.x(t["observed"]) - 6, panel.top + 14, f"observed {number(t['observed'], 3)}",
             "small", "end")  # fmt: skip
    panel.vertical(0.5, "axis")
    fig.bottom = panel.top + panel.height + 20
    return fig


def _scatter(fig: Figure, panel: Panel, poles: list[dict], read, head: str, labels: bool) -> None:
    """Each pole's 1 - r against the percentile that ``read`` gives.

    Args:
        fig: The figure.
        panel: The panel to draw in.
        poles: The pole rows.
        read: Gives a pole's percentile.
        head: The panel's name.
        labels: Whether to label the y axis.
    """
    panel.head(head)
    panel.grid_x([0, 0.5, 1], 1)
    panel.grid_y([0, 0.02, 0.04, 0.06, 0.08], 3, labels=labels)
    panel.x_title("percentile within the layer")
    for p in poles:
        colour = "c1" if p["keeps"] else "c2"
        _mark(fig, p["keeps"], panel.x(read(p)), panel.y(1 - p["pearson"]), 2.6,
              f"f{colour} half")  # fmt: skip


def move(rows: list[dict]) -> Figure:
    """The output-weight Fisher and the RMS percentile against how much the activation moves.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    poles = [r for r in rows if r["table"] == "pole"]
    plain = _test(rows, "one_minus_r", "spearman", "model", "output")
    partial = _test(rows, "one_minus_r", "partial_spearman", "model", "output")
    rms = _test(rows, "one_minus_r", "spearman", "rms", "activation")
    fig = Figure(
        ["Neurons that fire more move less, and the", "output Fisher adds nothing beyond that"],
        [
            "Left: each pole's output-weight Fisher percentile. Right: its activation RMS "
            "percentile. Both within the layer.",
            f"Spearman ρ with 1 − r: {number(plain['observed'], 3)} and "
            f"{number(rms['observed'], 3)}. "
            f"With the RMS taken out, the Fisher's ρ is {number(partial['observed'], 3)}.",
        ],
        f"Two scatter plots of 1 − r, the base-to-instruct activation change, for 311 poles. "
        f"Left, against the output-weight Fisher percentile: Spearman "
        f"{number(plain['observed'], 3)}. Right, against the activation RMS percentile: "
        f"{number(rms['observed'], 3)}. Partial Spearman of the Fisher with the RMS taken out: "
        f"{number(partial['observed'], 3)}.",
    )
    gap, left, right = 36, 60, 24
    width = (760 - left - right - gap) / 2
    panels = [Panel(fig, (left + i * (width + gap), fig.body_top + 34, width, 200),
                    (0, 1), (0, 0.08)) for i in range(2)]  # fmt: skip
    fig.motion += HALF
    reads = (lambda p: p["percentile"]["model"]["output"], lambda p: p["percentile"]["rms"])
    heads = ("output-weight Fisher", "activation RMS")
    for i, (panel, read, head) in enumerate(zip(panels, reads, heads, strict=True)):
        _scatter(fig, panel, poles, read, head, labels=i == 0)
    fig.text(14, panels[0].top - 8, "1 − r", "small")
    bottom = panels[0].top + panels[0].height + 52
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"keep.svg": keep(rows), "shuffles.svg": null(rows), "move.svg": move(rows)}
