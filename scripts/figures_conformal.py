"""The figures of papers/conformal-line, drawn from its results (#6417)."""

from figure_svg import Figure, Panel

TOP = 0.5  # the corpus panels' top; a pole above it is drawn on the top edge and counted
SHORT = {"EleutherAI/pythia-70m": "Pythia-70m", "openai-community/gpt2": "GPT-2",
         "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B"}  # fmt: skip
ORDER = list(SHORT)


def _poles(rows: list[dict]) -> list[dict]:
    """The pole rows.

    Args:
        rows: The results.

    Returns:
        The rows of the ``pole`` table.
    """
    return [r for r in rows if r.get("table") == "pole"]


def _median(values: list[float]) -> float:
    """The median of ``values``.

    Args:
        values: The numbers.

    Returns:
        Their median: the mean of the two middle values for an even count.
    """
    xs = sorted(values)
    n = len(xs)
    return (xs[(n - 1) // 2] + xs[n // 2]) / 2


def _strip(panel: Panel, poles: list[dict], fire) -> None:
    """One column of dots for each model, with its median as a short ink line and value.

    Args:
        panel: The panel.
        poles: The pole rows.
        fire: Each pole's share of held-out windows that fire.
    """
    for k, model in enumerate(ORDER):
        own = [fire(r) for r in poles if r["model"] == model]
        for i, value in enumerate(own):
            jitter = ((i * 37) % 21 - 10) / 10 * 0.28  # a fixed spread, so dots don't overlap
            panel.figure.dot(panel.x(k + jitter), panel.y(min(value, TOP)), 1.6, "fneutral soft")
        median = _median(own)
        panel.figure.line(panel.x(k - 0.36), panel.y(median), panel.x(k + 0.36),
                          panel.y(median), "contour")  # fmt: skip
        panel.figure.text(panel.x(k), panel.y(median) - 6, f"{median:.1%}", "small ink", "middle")
        panel.figure.text(panel.x(k), panel.top + panel.height + 15, SHORT[model], "small",
                          "middle")  # fmt: skip
        above = sum(v > TOP for v in own)
        if above:
            panel.figure.text(panel.x(k), panel.top - 4, f"{above} above", "small", "middle")


def fire(rows: list[dict]) -> Figure:
    """Each pole's false-fire share on held-out corpus windows, under each line.

    Args:
        rows: The results.

    Returns:
        The figure.
    """
    poles = _poles(rows)
    own = {m: _median([1 - r["own_cover"] for r in poles if r["model"] == m]) for m in ORDER}
    fig = Figure(
        [
            f"The fixed line fires on {min(own.values()):.1%} to {max(own.values()):.0%} of corpus "
            "windows, and the conformal line on about 5%"
        ],
        [
            "The share of 1,000 held-out 20-token corpus windows on which each VERIFIED pole",
            "fires. Left: the record's own line, 0.25 × its maximum. Right: the split-conformal",
            "line at α = 0.05. One dot per pole; the ink line and value are each model's median.",
        ],
        f"Two strip plots of {len(poles)} poles in four models. Under the record's line the "
        "medians are "
        + ", ".join(f"{own[m]:.1%} for {SHORT[m]}" for m in ORDER)
        + ". Under the conformal line every model's median is about 5%.",
    )
    for left, head, share in ((64, "Record's line (0.25 × max)", lambda r: 1 - r["own_cover"]),
                              (420, "Conformal line, α = 0.05",
                               lambda r: 1 - r["a0.05"]["cover"])):  # fmt: skip
        panel = Panel(fig, (left, fig.body_top + 44, 300, 240), (-0.5, 3.5), (0.0, TOP))
        fig.text(left, panel.top - 24, head, "head")  # above the "N above" counts
        panel.grid_y((0.0, 0.1, 0.2, 0.3, 0.4, 0.5), places=1)
        panel.horizontal(0.05, "dip")  # α, for reference
        _strip(panel, poles, share)
    fig.text(420 + 300 + 4, fig.body_top + 44 + 240 * (1 - 0.05 / TOP) + 4, "5%", "small")
    fig.bottom = fig.body_top + 44 + 240 + 30
    return fig


def pooled(poles: list[dict], model: str, key: str) -> float:
    """The share of one model's probes predicted quiet, over all its poles, that fire at ``key``.

    Args:
        poles: The pole rows.
        model: The model, or "all".
        key: ``own``, ``a0.05`` or ``a0.01``.

    Returns:
        The pooled share.
    """
    group = [r[key] for r in poles if model in ("all", r["model"])]
    return sum(g["quiet_fired"] for g in group) / sum(g["quiet"] for g in group)


def hard(rows: list[dict]) -> Figure:
    """The fire rate of probes predicted quiet under each conformal line, against its α.

    Args:
        rows: The results.

    Returns:
        The figure.
    """
    poles = _poles(rows)
    at05 = {m: pooled(poles, m, "a0.05") for m in ORDER}
    fig = Figure(
        ["Quiet probes fire far above the bound in Qwen2.5, and near it in Pythia-70m and GPT-2"],
        [
            "The share of each model's probes predicted quiet that fire under the conformal",
            "line, pooled over its poles, at α = 0.05 and α = 0.01. The dashed line is α itself:",
            "the rate the guarantee bounds on corpus text. Each model's line is labelled.",
        ],
        "A slope chart of four models. At α = 0.05 the probes predicted quiet fire at "
        + ", ".join(f"{at05[m]:.1%} in {SHORT[m]}" for m in ORDER)
        + ".",
    )
    panel = Panel(fig, (90, fig.body_top + 28, 420, 260), (-0.2, 1.2), (0.0, 0.3))
    panel.head("Probes predicted quiet that fire")
    panel.grid_y((0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3), places=2)
    for x, alpha in ((0, 0.05), (1, 0.01)):
        fig.text(panel.x(x), panel.top + panel.height + 15, f"α = {alpha}", "small", "middle")
    panel.curve([0, 1], [0.05, 0.01], "line sneutral dash")
    fig.text(panel.x(0) - 8, panel.y(0.05) + 4, "α", "small", "end")
    ends = []
    for model in ORDER:
        ys = [pooled(poles, model, "a0.05"), pooled(poles, model, "a0.01")]
        panel.curve([0, 1], ys, "curve")
        for x, y in zip((0, 1), ys, strict=True):
            fig.dot(panel.x(x), panel.y(y), 3.5, "fc1 ring")
        ends.append((panel.y(ys[1]), f"{SHORT[model]}: {ys[0]:.1%} → {ys[1]:.1%}"))
    last = -1e9
    for y, label in sorted(ends):  # top to bottom, 13 px apart so labels don't overlap
        y = max(y + 4, last + 13)
        fig.text(panel.x(1) + 18, y, label, "small")
        last = y
    fig.motion += ".dash{stroke-dasharray:6 4}"
    fig.bottom = panel.top + panel.height + 30
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """The paper's figures by file name.

    Args:
        rows: The results.

    Returns:
        The figures.
    """
    return {"fire.svg": fire(rows), "hard.svg": hard(rows)}
