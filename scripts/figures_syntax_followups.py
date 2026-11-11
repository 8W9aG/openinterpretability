"""The figures of papers/syntax-positive-controls/README.md from its follow-up rows.

The rows are those of ``results.followups.jsonl``, written by ``scripts/syntax_followups.py``.
"""

from figure_svg import Figure, Panel

DEPTHS = [1, 2, 3, 4, 5]
CHAIN_MIN_ITEMS = 4  # the fewest items behind a point of the close-later figure
CHAIN_SETS = (("clean", "c1"), ("L2 N676+ ablated", "c2"))


def _close_later_figure() -> Figure:
    """Return the empty figure of :func:`close_later`, with its titles and alt text."""
    return Figure(
        ["GPT-2 closes the rest of its brackets later, and not without L2 N676+"],
        [
            "Probability that the next token closes a round bracket, just after the model's most "
            "probable closer, with brackets still open.",
            f"A point needs at least {CHAIN_MIN_ITEMS} of the 8 written items to still have a "
            "bracket open. Hollow dots: the model, once every bracket is closed.",
        ],
        "A plot of closer probability against the number of brackets first opened. With "
        "brackets still open, the model gives 0.62 at depth 3, 0.55 at 4 and 0.36 at 5. Once "
        "all are closed it gives 0.001 at depth 3, and about 0.2 at depths 4 and 5. With L2 "
        "N676+ ablated it gives 0.02 or less at every depth.",
    )


def close_later(followups: list[dict]) -> Figure:
    """Draw the probability of one more closer after the first, with brackets still open."""
    found = {(r["set"], r["depth"]): r for r in followups if r["kind"] == "close_later"}
    fig = _close_later_figure()
    panel = Panel(fig, (64, fig.body_top + 22, 640, 220), (1.7, 5.3), (0, 0.7))
    panel.grid_x(DEPTHS[1:], 0)
    panel.grid_y([0, 0.2, 0.4, 0.6], 1)
    panel.x_title("brackets open before the first closer")
    for depth in DEPTHS[1:]:
        row = found[("clean", depth)]
        if row["all_closed_items"] >= CHAIN_MIN_ITEMS:
            fig.dot(panel.x(depth), panel.y(row["all_closed_p_closer"]), 6, "open sneutral")
    for name, colour in CHAIN_SETS:
        points = [
            (d, found[(name, d)]["still_open_p_closer"])
            for d in DEPTHS[1:]
            if found[(name, d)]["still_open_items"] >= CHAIN_MIN_ITEMS
        ]
        panel.curve([d for d, _ in points], [v for _, v in points], f"line s{colour}")
        for depth, value in points:
            fig.dot(panel.x(depth), panel.y(value), 3, f"f{colour}")
    bottom = panel.top + panel.height + 62
    legend = (
        ("fc1", "model, brackets still open"),
        ("fc2", "L2 N676+ ablated, brackets still open"),
        ("open sneutral", "model, all closed"),
    )
    fig.legend(bottom, legend)
    fig.bottom = bottom
    return fig
