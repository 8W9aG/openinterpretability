"""The figures of papers/random-change-keep/README.md, from its results.jsonl (#5154)."""

from figure_motion import counter
from figure_svg import Figure, Panel

# One colour and one shape per kind of change through the paper: circles are the instruct
# change, triangles the matched random change. The shape keeps them apart without colour.
KINDS = (("instruct", "c1", "instruct change"), ("random", "c2", "random change, same norm"))
COLOURS = {which: colour for which, colour, _ in KINDS}
# The rows of Figure 1, top to bottom: each condition and its label.
ROWS = (("instruct", "instruct, all weights"), ("random_all", "random, all weights"),
        ("instruct_mlp", "instruct, MLP only"), ("random_mlp", "random, MLP only"))  # fmt: skip
MODELS = (("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"), ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"))
GRID_MODEL = "Qwen/Qwen2.5-0.5B"  # Figure 2 shows the model with the most poles
GATES = {"specificity": 0.8, "sensitivity": 0.6}  # the review's gates (ReviewPolicy)
CYCLE = "10s"
COLUMNS = 38  # cells in each row of Figure 2


def kind(condition: str) -> str:
    """``instruct`` or ``random``: the kind of change that ``condition`` adds.

    Args:
        condition: A condition of ``scripts/random_delta_keep.py``.
    """
    return "random" if condition.startswith("random") else "instruct"


def _mark(fig: Figure, which: str, x: float, y: float, size: float, style: str) -> None:
    """One mark in the kind's shape: a circle for instruct, a triangle for random.

    Args:
        fig: The figure.
        which: ``instruct`` or ``random``.
        x: The mark's centre, in pixels.
        y: The mark's centre, in pixels.
        size: The mark's radius, in pixels.
        style: The mark's CSS classes.
    """
    if which == "instruct":
        fig.dot(x, y, size, style)
    else:
        h = size * 1.2
        fig.path([(x, y - h), (x + h, y + h * 0.8), (x - h, y + h * 0.8)], style, close=True)


def _legend(fig: Figure, y: float) -> None:
    """Each kind's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for which, colour, label in KINDS:
        _mark(fig, which, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, label)
        x += 30 + 6.4 * len(label)


def summaries(rows: list[dict], model: str, condition: str) -> list[dict]:
    """The summary rows of one model and condition, one for each seed, by seed.

    Args:
        rows: The paper's results.
        model: The base model's name.
        condition: The condition.
    """
    got = [r for r in rows if r["table"] == "summary" and r["model"] == model
           and r["condition"] == condition]  # fmt: skip
    return sorted(got, key=lambda r: r["seed"] or 0)


def margin(readings: dict) -> float:
    """How far a pole's base readings sit above the nearer of the two gates.

    Args:
        readings: A pole's readings, with its specificity and sensitivity.
    """
    return min(readings[gate] - line for gate, line in GATES.items())


def keep(rows: list[dict]) -> Figure:
    """The keep share of each condition and seed, for both models.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    first = summaries(rows, MODELS[0][0], "instruct")[0]
    randoms = summaries(rows, MODELS[0][0], "random_all")
    fig = Figure(
        ["On Qwen2.5-0.5B, a random change of the instruct change's",
         "size breaks as many explanations, but on 1.5B far fewer"],
        ["Each mark is the share of passing explanations that still pass their tests, with a",
         "95% interval. Each random change has the norm of the instruct change in every matrix."],
        f"Two panels, one for each Qwen2.5 model. Each has four rows: the instruct change and "
        f"a random change of its size, on all weights and on the MLP weights only. On "
        f"Qwen2.5-0.5B, {first['keeps']} of {first['reproduced_on_base']} explanations keep "
        f"under the instruct change, and {min(r['keeps'] for r in randoms)} to "
        f"{max(r['keeps'] for r in randoms)} under random changes of all weights.",
    )  # fmt: skip
    top, height = fig.body_top + 30, 150
    for slot, (model, label) in enumerate(MODELS):
        panel = Panel(fig, (190 + slot * 290, top, 250, height), (0.6, 1.0), (0, len(ROWS)))
        panel.grid_x([0.6, 0.7, 0.8, 0.9, 1.0], 1)
        panel.head(label)
        panel.x_title("share of explanations that keep")
        for i, (condition, name) in enumerate(ROWS):
            y = panel.y(len(ROWS) - i - 0.5)
            if slot == 0:
                fig.text(panel.left - 14, y + 4, name, "small", "end")
            which = kind(condition)
            got = summaries(rows, model, condition)
            for j, r in enumerate(got):
                dy = (j - (len(got) - 1) / 2) * 6  # each seed on its own line, in seed order
                low, high = r["keep_ci95"]
                colour = COLOURS[which]
                fig.line(panel.x(low), y + dy, panel.x(high), y + dy, f"range s{colour} soft")
                _mark(fig, which, panel.x(r["keep_share"]), y + dy, 3.6, f"f{colour} ring")
    bottom = top + height + 50
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def pole_order(rows: list[dict], model: str) -> list[str]:
    """The model's poles that reproduce on the base, nearest a gate first.

    Args:
        rows: The paper's results.
        model: The base model's name.
    """
    base = [r for r in rows if r["table"] == "base" and r["model"] == model
            and r["readings"]["passes"]]  # fmt: skip
    return [r["slug"] for r in sorted(base, key=lambda r: (margin(r["readings"]), r["slug"]))]


def flipped(rows: list[dict], model: str, condition: str, seed) -> set[str]:
    """The poles that reproduce on the base and fail under one condition and seed.

    Args:
        rows: The paper's results.
        model: The base model's name.
        condition: The condition.
        seed: The condition's seed, or ``None``.
    """
    return {r["slug"] for r in rows if r["table"] == "pole" and r["model"] == model
            and r["condition"] == condition and r["seed"] == seed
            and r["base_passes"] and not r["keeps"]}  # fmt: skip


CELL, GAP = 15.0, 3.0  # Figure 2's cell size and the space between cells, in pixels


def _cell(i: int, left: float, top: float) -> tuple[float, float]:
    """The top left corner of cell ``i`` of Figure 2, in rows of ``COLUMNS``.

    Args:
        i: The cell's place in the order of the poles.
        left: The grid's left edge, in pixels.
        top: The grid's top edge, in pixels.
    """
    return left + (i % COLUMNS) * (CELL + GAP), top + (i // COLUMNS) * (CELL + GAP)


def flips(rows: list[dict]) -> Figure:
    """Which poles flip under each change on all weights, one frame per change.

    Args:
        rows: The paper's results.

    Returns:
        The figure. It steps through the changes and rests on the last.
    """
    order = pole_order(rows, GRID_MODEL)
    frames = [("instruct", None)] + [
        ("random_all", r["seed"]) for r in summaries(rows, GRID_MODEL, "random_all")
    ]
    sets = [flipped(rows, GRID_MODEL, c, s) for c, s in frames]
    near = len(order) // 4
    in_near = [len(s & set(order[:near])) for s in sets]
    fig = Figure(
        ["On Qwen2.5-0.5B, instruction tuning and random changes",
         "flip many of the same poles: those near a gate"],
        [f"Each cell is one of Qwen2.5-0.5B's {len(order)} passing poles, nearest a gate first. "
         "A filled cell flips:",
         "it passes its test prompts on the base model and fails on the changed one."],
        f"A grid of {len(order)} cells, in rows of {COLUMNS}, ordered by how far each pole's "
        f"base readings sit above the nearer gate. The frames step through the instruct change "
        f"and {len(frames) - 1} random changes of all weights. Each flips {min(map(len, sets))} "
        f"to {max(map(len, sets))} poles, and {min(in_near)} to {max(in_near)} of them lie in "
        f"the first quarter of the grid.",
    )  # fmt: skip
    left, top = 40.0, fig.body_top + 34
    for i in range(len(order)):
        fig.bar(*_cell(i, left, top), CELL, CELL, "fwash")
    starts = [100 * i / len(frames) for i in range(len(frames))]
    for (cls, css), (condition, seed), got in zip(counter("frame", starts, CYCLE), frames, sets,
                                                  strict=True):  # fmt: skip
        fig.motion += css
        which = kind(condition)
        name = "instruct change" if which == "instruct" else f"random change, seed {seed}"
        fig.text(left, top - 12, f"{name}: {len(got)} flip", f"head {cls}")
        for i, slug in enumerate(order):
            if slug in got:
                x, y = _cell(i, left, top)
                _mark(fig, which, x + CELL / 2, y + CELL / 2, 5.2, f"f{COLOURS[which]} {cls}")
    bottom = _cell(len(order) - 1, left, top)[1] + CELL + GAP + 20
    fig.text(left, bottom, "Poles run left to right, then down: the nearest to a gate at the "
             "top left, the furthest at the bottom.", "small")  # fmt: skip
    _legend(fig, bottom + 30)
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
