"""The figures of papers/rome-keep/README.md, from its results.jsonl (#5167).

The shapes and the grid are those of papers/random-change-keep (figures_random_change_keep).
"""

from figure_motion import counter
from figure_svg import Figure, Panel
from figures_random_change_keep import CELL, COLUMNS, CYCLE, GAP, _cell, _mark, margin

# A ROME edit is a circle in the first colour, and a random rank-one edit a triangle in the
# second. The shape keeps them apart without colour.
SHAPES = {"rome": ("instruct", "c1"), "random": ("random", "c2")}
LABELS = (("rome", "ROME edit"), ("random", "random rank-one edit, same norm"))
LAYERS = 6


def _legend(fig: Figure, y: float) -> None:
    """Each edit's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for name, label in LABELS:
        which, colour = SHAPES[name]
        _mark(fig, which, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, label)
        x += 30 + 6.4 * len(label)


def keep_by_layer(rows: list[dict]) -> dict[str, list[float]]:
    """Each kind's keep share at each layer, over every edit and the poles that reproduce.

    Args:
        rows: The paper's results.
    """
    out = {}
    for kind in SHAPES:
        shares = []
        for layer in range(LAYERS):
            got = [r["keeps"] for r in rows if r["table"] == "pole" and r["condition"] == kind
                   and r["layer"] == layer and r["base_passes"]]  # fmt: skip
            shares.append(sum(got) / len(got))
        out[kind] = shares
    return out


def keep(rows: list[dict]) -> Figure:
    """The keep share at each layer, for ROME and random edits at layer 2.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    shares = keep_by_layer(rows)
    edit = next(r["edit_layer"] for r in rows if r["table"] == "summary")
    fig = Figure(
        ["A ROME edit at layer 2 leaves every pole below it intact,",
         "and breaks fewer above it than a random edit of its size"],
        ["Each mark is the share of pythia-70m's VERIFIED explanations at that layer that",
         "still pass their probes, over 10 edits of unrelated facts and 10 random edits."],
        f"Keep share by layer 0 to 5. At layers 0 to {edit} both kinds keep every explanation. "
        f"Above, ROME keeps {min(shares['rome'][edit + 1:]):.0%} to "
        f"{max(shares['rome'][edit + 1:]):.0%} and random edits "
        f"{min(shares['random'][edit + 1:]):.0%} to {max(shares['random'][edit + 1:]):.0%}.",
    )  # fmt: skip
    panel = Panel(fig, (80, fig.body_top + 30, 620, 200), (-0.5, LAYERS - 0.5), (0.85, 1.005))
    panel.grid_y([0.85, 0.9, 0.95, 1.0], 2)
    for layer in range(LAYERS):
        fig.text(panel.x(layer), panel.top + panel.height + 15, f"L{layer}", "small", "middle")
    panel.x_title("layer of the pole")
    panel.vertical(edit + 0.5, "dip")
    fig.text(panel.x(edit + 0.5) + 6, panel.top + 12, "edit at layer 2: below it, no input changes",
             "small")  # fmt: skip
    for kind, (which, colour) in SHAPES.items():
        xs = [layer + (-0.12 if kind == "rome" else 0.12) for layer in range(LAYERS)]
        panel.curve(xs, shares[kind], f"line s{colour}")
        for x, share in zip(xs, shares[kind], strict=True):
            _mark(fig, which, panel.x(x), panel.y(share), 4.5, f"f{colour} ring")
    bottom = panel.top + panel.height + 50
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def grid_order(rows: list[dict]) -> list[str]:
    """The poles that reproduce, by layer, then nearest a gate first within each layer.

    Args:
        rows: The paper's results.
    """
    base = {r["slug"]: r for r in rows if r["table"] == "base" and r["readings"]["passes"]}
    return sorted(base, key=lambda s: (base[s]["layer"], margin(base[s]["readings"]), s))


def flips(rows: list[dict]) -> Figure:
    """Which poles flip under each fact's ROME edit and its random edit, one frame per fact.

    Args:
        rows: The paper's results.

    Returns:
        The figure. It steps through the facts and rests on the last.
    """
    order = grid_order(rows)
    facts = sorted({r["fact"] for r in rows if r["table"] == "summary"})
    flipped = {(r["condition"], r["fact"], r["slug"]) for r in rows if r["table"] == "pole"
               and r["base_passes"] and not r["keeps"]}  # fmt: skip
    layer_of = {r["slug"]: r["layer"] for r in rows if r["table"] == "base"}
    low = sum(layer_of[s] <= 2 for s in order)
    fig = Figure(
        ["Each edit flips a few poles above layer 2,", "and none at or below it"],
        [f"Each cell is one of pythia-70m's {len(order)} VERIFIED poles, by layer, nearest a gate "
         "first within each.",
         f"The first {low} cells are layers 0 to 2. Circles flip under ROME, triangles under the "
         "random edit."],
        f"A grid of {len(order)} cells, in rows of {COLUMNS}. The frames step through 10 facts. "
        f"No cell among the first {low} ever flips. Above them, each edit flips a few cells.",
    )  # fmt: skip
    left, top = 40.0, fig.body_top + 34
    for i in range(len(order)):
        fig.bar(*_cell(i, left, top), CELL, CELL, "fneutral soft" if i < low else "fwash")
    starts = [100 * i / len(facts) for i in range(len(facts))]
    for (cls, css), fact in zip(counter("frame", starts, CYCLE), facts, strict=True):
        fig.motion += css
        counts = {k: sum((k, fact, s) in flipped for s in order) for k in SHAPES}
        fig.text(left, top - 12, f"fact {fact + 1}: ROME flips {counts['rome']}, random "
                 f"{counts['random']}", f"head {cls}")  # fmt: skip
        for i, slug in enumerate(order):
            x, y = _cell(i, left, top)
            for kind, (which, colour) in SHAPES.items():
                if (kind, fact, slug) in flipped:
                    dx = -3 if kind == "rome" else 3
                    _mark(fig, which, x + CELL / 2 + dx, y + CELL / 2, 4.0, f"f{colour} {cls}")
    bottom = _cell(len(order) - 1, left, top)[1] + CELL + GAP + 20
    fig.text(left, bottom, "Shaded cells are layers 0 to 2, which the edit cannot change.",
             "small")  # fmt: skip
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
