"""The figures of papers/rome-sequence/README.md, from its results.jsonl (#5362).

A ROME sequence is a circle in the first colour, and the random sequence a triangle in the
second, as in papers/rome-keep (figures_rome_keep).
"""

import math

from figure_motion import reveal
from figure_svg import Figure, Panel
from figures_random_change_keep import _mark
from rome_sequence_stats import hazard

CYCLE = "10s"
SHAPES = {"rome": ("instruct", "c1"), "random": ("random", "c2")}
LABELS = (("rome", "ROME edits, in turn"), ("random", "random rank-one edits, same norms"))


def _legend(fig: Figure, y: float) -> None:
    """Each sequence's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for name, label in LABELS:
        which, colour = SHAPES[name]
        _mark(fig, which, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, label, "small")
        x += 30 + 5.8 * len(label)
    fig.bottom = y


def _checkpoints(rows: list[dict], kind: str) -> list[dict]:
    """One sequence's checkpoint rows, in order of edits.

    Args:
        rows: The paper's results.
        kind: ``rome`` or ``random``.
    """
    return sorted((r for r in rows if r["table"] == "checkpoint" and r["condition"] == kind),
                  key=lambda r: r["edits"])  # fmt: skip


def decay(rows: list[dict]) -> Figure:
    """The poles above the edit layer that keep, against the weight distance, edit by edit.

    Args:
        rows: The paper's results.

    Returns:
        The figure. Its marks appear checkpoint by checkpoint, with the fitted curves.
    """
    seqs = {k: _checkpoints(rows, k) for k in SHAPES}
    n0 = seqs["rome"][0]["poles_above"]
    far = max(c["distance"] for s in seqs.values() for c in s)
    fits = {k: hazard(v)["h"] for k, v in seqs.items()}
    fig = Figure(
        ["As ROME edits pile up, the explanations above the layer fade steadily,",
         "with no threshold, and more slowly than under random edits of the same size"],
        [f"Each mark is a checkpoint: the {n0} reviewed poles above layer 2 that still pass, "
         "against ‖ΔW‖/‖W₀‖.",
         "The curves are N·exp(−h·distance), fitted to each sequence."],
        f"Keeps fall from {n0} to {seqs['rome'][-1]['keeps_above']} under ROME and to "
        f"{seqs['random'][-1]['keeps_above']} under random edits, at a distance of about "
        f"{far:.0f}. The fitted rates are {fits['rome']:.2f} and {fits['random']:.2f}.",
    )  # fmt: skip
    panel = Panel(fig, (80, fig.body_top + 30, 600, 240), (0, far * 1.03), (0, n0 * 1.05))
    panel.grid_x([0, 2, 4, 6, 8, 10, 12], 0)
    panel.grid_y([0, 20, 40, 60, 80], 0)
    panel.x_title("weight distance of layer 2, ‖W_n − W₀‖ / ‖W₀‖")
    for kind, (_, colour) in SHAPES.items():
        xs = [far * i / 60 for i in range(61)]
        panel.curve(xs, [n0 * math.exp(-fits[kind] * x) for x in xs], f"line s{colour}")
    counts = [c["edits"] for c in seqs["rome"]]
    for i in range(len(counts)):
        shown = f"step{i}"
        fig.motion += reveal(shown, 4 + 88 * i / len(counts), CYCLE)
        for kind, (which, colour) in SHAPES.items():
            c = seqs[kind][i]
            _mark(fig, which, panel.x(c["distance"]), panel.y(c["keeps_above"]), 4.0,
                  f"f{colour} ring {shown}")  # fmt: skip
    last = seqs["rome"][-1]
    fig.text(panel.x(last["distance"]) - 6, panel.y(last["keeps_above"]) - 10,
             f"{last['edits']} edits", f"small step{len(counts) - 1}", "end")  # fmt: skip
    _legend(fig, panel.top + panel.height + 52)
    return fig


def edits(rows: list[dict]) -> Figure:
    """Each edit's norm, and the loss after each checkpoint, over the sequence.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    norms = [r["edit_norm"] for r in sorted((r for r in rows if r["table"] == "fact"),
                                            key=lambda r: r["index"])]  # fmt: skip
    seqs = {k: _checkpoints(rows, k) for k in SHAPES}
    n = len(norms)
    fig = Figure(
        ["Later edits are larger, and the loss rises with them"],
        ["Left: the Frobenius norm of each ROME edit, in the order made. Right: the "
         "language-model loss on",
         "20,000 corpus tokens after each checkpoint, for both sequences."],
        f"Left, a bar chart of {n} edit norms, from {norms[0]:.1f} for the first to "
        f"{max(norms):.1f} at most. Right, the loss rises from {seqs['rome'][0]['loss']:.2f} "
        f"to {seqs['rome'][-1]['loss']:.2f} under ROME and {seqs['random'][-1]['loss']:.2f} "
        "under random edits.",
    )  # fmt: skip
    top = fig.body_top + 34
    left = Panel(fig, (60, top, 300, 200), (0, n + 1), (0, max(norms) * 1.08))
    left.grid_y([0, 10, 20, 30, 40], 0)
    left.grid_x([1, 25, 50, 75, 100], 0)
    left.x_title("edit")
    left.head("norm of each ROME edit")
    for i, v in enumerate(norms):
        fig.bar(left.x(i + 0.6), left.y(v), left.x(1.4) - left.x(0.6), left.y(0) - left.y(v),
                "fc1")  # fmt: skip
    high = max(c["loss"] for s in seqs.values() for c in s)
    right = Panel(fig, (420, top, 300, 200), (0, n), (0, high * 1.08))
    right.grid_y([0, 5, 10, 15], 0)
    right.grid_x([0, 25, 50, 75, 100], 0)
    right.x_title("edits")
    right.head("loss on the corpus sample")
    for kind, (which, colour) in SHAPES.items():
        right.curve([c["edits"] for c in seqs[kind]], [c["loss"] for c in seqs[kind]],
                    f"line s{colour}")  # fmt: skip
        for c in seqs[kind]:
            _mark(fig, which, right.x(c["edits"]), right.y(c["loss"]), 3.5, f"f{colour} ring")
    _legend(fig, top + 200 + 52)
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"decay.svg": decay(rows), "edits.svg": edits(rows)}
