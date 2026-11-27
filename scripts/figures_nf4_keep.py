"""The figures of papers/nf4-keep/README.md, from its results.jsonl (#5158).

The shapes and the grid are those of papers/random-change-keep (figures_random_change_keep).
"""

from figure_motion import counter
from figure_svg import Figure, Panel
from figures_random_change_keep import (
    CELL,
    COLUMNS,
    CYCLE,
    GAP,
    _cell,
    _mark,
    flipped,
    pole_order,
    summaries,
)

# NF4 is a circle in the first colour, and its matched random change a triangle in the second.
SHAPES = {"nf4": ("instruct", "c1"), "random_nf4": ("random", "c2"), "bf16": ("instruct", "c3")}
LABELS = (("nf4", "NF4 rounding"), ("random_nf4", "random change, NF4's norm"),
          ("bf16", "bfloat16 rounding"))  # fmt: skip
MODELS = (("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"), ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"))
GRID_MODEL = "Qwen/Qwen2.5-0.5B"  # Figure 2 shows the model with the most poles


def _legend(fig: Figure, y: float, names=("nf4", "random_nf4", "bf16")) -> None:
    """Each condition's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
        names: The conditions to show.
    """
    x = 20.0
    for name, label in LABELS:
        if name in names:
            which, colour = SHAPES[name]
            _mark(fig, which, x + 5, y - 4, 5, f"f{colour}")
            fig.text(x + 15, y, label)
            x += 30 + 6.4 * len(label)


def keep(rows: list[dict]) -> Figure:
    """The keep share of each condition and seed, for both models.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    small = summaries(rows, MODELS[1][0], "nf4")[0]
    randoms = summaries(rows, MODELS[1][0], "random_nf4")
    fig = Figure(
        ["NF4 rounding keeps most explanations, and on Qwen2.5-1.5B",
         "more than a random change of the same size"],
        ["Each mark is the share of passing explanations that still pass their tests, with a",
         "95% interval. Each random change has the norm of NF4's error in every matrix."],
        f"Two panels, one for each Qwen2.5 model, with three rows: NF4 rounding, random "
        f"changes of the same norm, and bfloat16 rounding, which changes nothing. On "
        f"Qwen2.5-1.5B, {small['keeps']} of {small['reproduced_on_base']} keep under NF4, and "
        f"{min(r['keeps'] for r in randoms)} to {max(r['keeps'] for r in randoms)} under the "
        f"random changes.",
    )  # fmt: skip
    top, height = fig.body_top + 30, 120
    names = [n for n, _ in LABELS]
    for slot, (model, label) in enumerate(MODELS):
        panel = Panel(fig, (190 + slot * 290, top, 250, height), (0.6, 1.0), (0, len(names)))
        panel.grid_x([0.6, 0.7, 0.8, 0.9, 1.0], 1)
        panel.head(label)
        panel.x_title("share of explanations that keep")
        for i, (condition, name) in enumerate(LABELS):
            y = panel.y(len(names) - i - 0.5)
            if slot == 0:
                fig.text(panel.left - 14, y + 4, name, "small", "end")
            which, colour = SHAPES[condition]
            got = summaries(rows, model, condition)
            for j, r in enumerate(got):
                dy = (j - (len(got) - 1) / 2) * 6  # each seed on its own line, in seed order
                low, high = r["keep_ci95"]
                fig.line(panel.x(low), y + dy, panel.x(high), y + dy, f"range s{colour} soft")
                _mark(fig, which, panel.x(r["keep_share"]), y + dy, 3.6, f"f{colour} ring")
    bottom = top + height + 50
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def flips(rows: list[dict]) -> Figure:
    """Which poles flip under NF4 and under each random change, one frame per change.

    Args:
        rows: The paper's results.

    Returns:
        The figure. It steps through the changes and rests on the last.
    """
    order = pole_order(rows, GRID_MODEL)
    frames = [("nf4", None)] + [("random_nf4", r["seed"])
                                for r in summaries(rows, GRID_MODEL, "random_nf4")]  # fmt: skip
    sets = [flipped(rows, GRID_MODEL, c, s) for c, s in frames]
    near = set(order[: len(order) // 4])
    fig = Figure(
        ["On Qwen2.5-0.5B, NF4 and random changes of its size",
         "flip many of the same poles: those near a gate"],
        [f"Each cell is one of Qwen2.5-0.5B's {len(order)} passing poles, nearest a gate first. "
         "A filled cell flips:",
         "it passes its test prompts on the base model and fails on the changed one."],
        f"A grid of {len(order)} cells, in rows of {COLUMNS}, ordered by how far each pole's "
        f"base readings sit above the nearer gate. The frames step through NF4 rounding and "
        f"{len(frames) - 1} random changes of its norm. Each flips {min(map(len, sets))} to "
        f"{max(map(len, sets))} poles, and {min(len(s & near) for s in sets)} to "
        f"{max(len(s & near) for s in sets)} of them lie in the first quarter of the grid.",
    )  # fmt: skip
    left, top = 40.0, fig.body_top + 34
    for i in range(len(order)):
        fig.bar(*_cell(i, left, top), CELL, CELL, "fwash")
    starts = [100 * i / len(frames) for i in range(len(frames))]
    for (cls, css), (condition, seed), got in zip(counter("frame", starts, CYCLE), frames, sets,
                                                  strict=True):  # fmt: skip
        fig.motion += css
        which, colour = SHAPES[condition]
        name = "NF4 rounding" if seed is None else f"random change, seed {seed}"
        fig.text(left, top - 12, f"{name}: {len(got)} flip", f"head {cls}")
        for i, slug in enumerate(order):
            if slug in got:
                x, y = _cell(i, left, top)
                _mark(fig, which, x + CELL / 2, y + CELL / 2, 5.2, f"f{colour} {cls}")
    bottom = _cell(len(order) - 1, left, top)[1] + CELL + GAP + 20
    fig.text(left, bottom, "Poles run left to right, then down: the nearest to a gate at the "
             "top left, the furthest at the bottom.", "small")  # fmt: skip
    _legend(fig, bottom + 30, ("nf4", "random_nf4"))
    fig.bottom = bottom + 30
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"keep.svg": keep(rows), "flips.svg": flips(rows)}
