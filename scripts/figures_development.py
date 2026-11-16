"""The figure of papers/neuron-development, drawn from its results (issue #3376)."""

import math

from figure_svg import Figure, Panel

TICKS = (1, 10, 100, 1000, 10_000, 100_000)


def _x(step: int) -> float:
    """A step on a log axis. Step 0 sits at 1, the left end."""
    return math.log10(step + 1)


def _label(step: int) -> str:
    return f"{step // 1000}k" if step >= 1000 else str(step)


def _panels(fig: Figure) -> tuple[Panel, Panel]:
    """The correlation panel and the loss panel, with their grids, labels and the 16k line."""
    span = (0.0, _x(150_000))
    left = Panel(fig, (64, fig.body_top + 28, 400, 260), span, (-0.2, 1.0))
    right = Panel(fig, (520, fig.body_top + 28, 216, 260), span, (3.0, 11.5))
    for panel, head in ((left, "Correlation with the final form"), (right, "Loss (nats)")):
        panel.head(head)
        for tick in TICKS:
            panel.figure.line(panel.x(_x(tick)), panel.top, panel.x(_x(tick)),
                              panel.top + panel.height, "grid")  # fmt: skip
            panel.figure.text(panel.x(_x(tick)), panel.top + panel.height + 15, _label(tick),
                              "small", "middle")  # fmt: skip
        panel.x_title("training step (log scale)")
    left.grid_y((0.0, 0.5, 1.0), places=1)
    right.grid_y((4, 6, 8, 10), places=0)
    left.vertical(_x(16_000), "dip")
    right.vertical(_x(16_000), "dip")
    fig.text(right.x(_x(16_000)) - 6, right.y(10.8), "step 16k: loss", "small", "end")
    fig.text(right.x(_x(16_000)) - 6, right.y(10.8) + 14, "within 1% of final", "small", "end")
    return left, right


def development(rows: list[dict]) -> Figure:
    """Each pole's correlation with its final form by step, next to the loss."""
    poles = [r for r in rows if r["table"] == "pole"]
    loss = [r for r in rows if r["table"] == "loss"]
    fig = Figure(
        ["Training loss settles by step 16,000, but the VERIFIED poles keep changing"],
        [
            "Left: each of 9 Pythia-70m poles' correlation with its final activations, by step.",
            "A filled dot is a step where the pole's recorded probes pass its review's gates.",
            "Right: next-token loss on the same 300 windows.",
        ],
        "Nine lines of correlation with the final checkpoint that rise slowly to 1 by step "
        "143,000, beside a loss curve that is flat from about step 16,000.",
    )
    left, right = _panels(fig)
    for slug in sorted({r["slug"] for r in poles}):
        own = sorted((r for r in poles if r["slug"] == slug), key=lambda r: r["step"])
        left.curve([_x(r["step"]) for r in own], [r["r_final"] for r in own], "line sneutral")
        for r in own:
            # A filled dot's surface ring takes 1 px of its edge, so it gets 1 px more radius.
            cls, radius = ("fc1 ring", 5) if r["holds"] else ("open sneutral", 4)
            fig.dot(left.x(_x(r["step"])), left.y(r["r_final"]), radius, cls)
    loss = sorted(loss, key=lambda r: r["step"])
    right.curve([_x(r["step"]) for r in loss], [r["loss"] for r in loss], "curve")
    fig.legend(left.top + left.height + 50, [("fc1", "probes pass"), ("open sneutral", "fail")])
    fig.bottom = left.top + left.height + 52
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """The paper's figures by file name."""
    return {"development.svg": development(rows)}
