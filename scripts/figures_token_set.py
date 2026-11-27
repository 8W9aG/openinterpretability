"""The figures of papers/token-set-ceiling, drawn from its results (#5144)."""

from figure_svg import Figure, Panel, columns

SHORT = {"EleutherAI/pythia-70m": "Pythia-70m", "openai-community/gpt2": "GPT-2",
         "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B"}  # fmt: skip
BANDS = ("early", "middle", "late")
LABELS = ("token-set", "mixed", "context")
TOP = 0.7  # the share axis's top: the highest interval ends near 0.66


def _table(rows: list[dict], name: str) -> list[dict]:
    """The rows of one table.

    Args:
        rows: The results.
        name: The table, such as ``share``.

    Returns:
        Its rows, in file order.
    """
    return [r for r in rows if r.get("table") == name]


def share(rows: list[dict]) -> Figure:
    """The token-set share in each third of each model, with its interval and the overall line.

    Args:
        rows: The results.

    Returns:
        The figure.
    """
    shares = {(r["model"], r["depth"]): r for r in _table(rows, "share")}
    overall = shares[("all", "all")]
    early = [shares[(m, "early")]["token_set_share"] for m in SHORT]
    fig = Figure(
        [
            f"Only {overall['token_set_share']:.0%} of passing explanations are token sets, "
            "and most are in the first third"
        ],
        [
            "The share of each model's passing poles whose explanation is a token set, in the",
            "early, middle and late third of its layers. Each dot has a 95% Wilson interval and",
            "its count. The grey line is the share over all four models, labelled 'all'.",
        ],
        f"Four panels, one per model. The overall share is {overall['token_set_share']:.1%} "
        f"of {overall['poles']} poles. In the early third the share is "
        + ", ".join(f"{v:.0%} for {SHORT[m]}" for m, v in zip(SHORT, early, strict=True))
        + ". In the middle and late thirds it is 0% to 14%.",
    )
    panels = columns(fig, 4, 64, fig.body_top + 34, 230, (-0.5, 2.5), (0.0, TOP), gap=22)
    for k, (panel, model) in enumerate(zip(panels, SHORT, strict=True)):
        panel.head(SHORT[model])
        panel.grid_y((0.0, 0.2, 0.4, 0.6), places=1, labels=k == 0)
        panel.horizontal(overall["token_set_share"], "dip")
        for x, band in enumerate(BANDS):
            row = shares[(model, band)]
            low, high = row["ci95"]
            fig.line(panel.x(x), panel.y(low), panel.x(x), panel.y(high), "range sneutral")
            fig.dot(panel.x(x), panel.y(row["token_set_share"]), 4, "fc1 ring")
            fig.text(panel.x(x) + 7, panel.y(row["token_set_share"]) + 4,
                     f"{row['token-set']}/{row['poles']}", "small")  # fmt: skip
            fig.text(panel.x(x), panel.top + panel.height + 15, band, "small", "middle")
    fig.text(panels[2].left + panels[2].width, panels[2].y(overall["token_set_share"]) - 5,
             f"all: {overall['token_set_share']:.1%}", "small", "end")  # fmt: skip
    return fig


def token_r2(rows: list[dict]) -> Figure:
    """Each pole's ``token_r2`` by its label, with each label's median.

    Args:
        rows: The results.

    Returns:
        The figure.
    """
    poles = _table(rows, "pole")
    (check,) = _table(rows, "token_r2")
    low, high = check["ci95"]
    fig = Figure(
        [
            "The token mean explains far more of a token-set pole's activity "
            f"(AUROC {check['auroc']:.2f})"
        ],
        [
            "Each passing pole's token_r2, the share of its activation variance that a mean per",
            "token id explains on its scan corpus, by the label of its explanation. One dot per",
            "pole; the ink line and value are each label's median. The AUROC's 95% interval",
            f"over a bootstrap of poles is {low:.2f}–{high:.2f}.",
        ],
        f"A strip plot of {len(poles)} poles in three columns. The median token_r2 is "
        + ", ".join(
            f"{_median([p['token_r2'] for p in poles if p['label'] == lab]):.2f} for {lab}"
            for lab in LABELS
        )
        + ".",
    )
    panel = Panel(fig, (90, fig.body_top + 20, 560, 260), (-0.5, 2.5), (0.0, 1.0))
    panel.grid_y((0.0, 0.25, 0.5, 0.75, 1.0), places=2)
    for k, lab in enumerate(LABELS):
        own = [p["token_r2"] for p in poles if p["label"] == lab]
        for i, value in enumerate(own):
            jitter = ((i * 37) % 21 - 10) / 10 * 0.3  # a fixed spread, so dots don't overlap
            fig.dot(panel.x(k + jitter), panel.y(min(max(value, 0.0), 1.0)), 1.6, "fneutral soft")
        median = _median(own)
        fig.line(panel.x(k - 0.38), panel.y(median), panel.x(k + 0.38), panel.y(median), "contour")
        fig.text(panel.x(k + 0.42), panel.y(median) + 4, f"{median:.2f}", "small ink")
        fig.text(panel.x(k), panel.top + panel.height + 15, f"{lab} ({len(own)})", "small",
                 "middle")  # fmt: skip
    panel.x_title("Label of the explanation")
    return fig


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


def build(rows: list[dict]) -> dict[str, Figure]:
    """The paper's figures by file name.

    Args:
        rows: The results.

    Returns:
        ``{name: figure}``.
    """
    return {"share.svg": share(rows), "token-r2.svg": token_r2(rows)}
