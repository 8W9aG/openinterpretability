"""The figures of papers/wrapper-prune/README.md, from its results.jsonl (#5163).

The shapes and the grid are those of papers/random-change-keep (figures_random_change_keep).
Figures 1 and 2 read the paper's own run. Figure 3 compares it with the variants of #7071,
counted on the paper's poles.
"""

from figure_motion import counter
from figure_svg import Figure, Panel
from figures_random_change_keep import CELL, CYCLE, GAP, _cell, _mark, pole_order
from wrapper_prune_stats import changes, paper_poles_only, pole_sets, variant_rows

MODEL = "Qwen/Qwen2.5-0.5B"
# Top pruning is a circle in the first colour, and random pruning a triangle in the second.
SHAPES = {"top": ("instruct", "c1"), "random": ("random", "c2")}
LABELS = (("top", "top K by gradient × weight"), ("random", "K random, same layers"))


def _legend(fig: Figure, y: float) -> None:
    """Each prune's shape and name, in one row.

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


def recovery(rows: list[dict]) -> Figure:
    """The poles recovered and lost at each K, for top and random pruning.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    table = changes(rows)
    ks = sorted({r["k"] for r in table})
    top = {r["k"]: r for r in table if r["condition"] == "top"}
    failed, kept = table[0]["failed_on_instruct"], table[0]["kept_on_instruct"]
    back = [r["recovered"] for r in top.values()]
    fig = Figure(
        ["Pruning by gradient times weight brings back a few failed",
         "explanations, and breaks many more"],
        [f"Qwen2.5-0.5B's instruct model, with K MLP neurons' writes pruned. Left: of the {failed}",
         f"explanations that fail on it, those that pass again. Right: of the {kept} that keep, "
         "those that fail."],
        f"Two panels against K, from {ks[0]} to {ks[-1]}. Top pruning recovers "
        f"{min(back)} to {max(back)} "
        f"explanations and loses {top[ks[0]]['lost']} to {top[ks[-1]]['lost']}. Random pruning of "
        f"as many neurons recovers and loses few.",
    )  # fmt: skip
    top_px = fig.body_top + 30
    for slot, (field, total, name) in enumerate((("recovered", failed, "recovered"),
                                                 ("lost", kept, "lost"))):  # fmt: skip
        span = 20 if field == "recovered" else 160
        panel = Panel(fig, (70 + slot * 350, top_px, 290, 170), (-0.5, len(ks) - 0.5), (0, span))
        panel.grid_y([0, span / 4, span / 2, 3 * span / 4, span], 0)
        panel.head(f"{name}, of {total}")
        for i, k in enumerate(ks):
            fig.text(panel.x(i), panel.top + panel.height + 15, str(k), "small", "middle")
        panel.x_title("K, the neurons pruned")
        for condition, (which, colour) in SHAPES.items():
            got = [r for r in table if r["condition"] == condition]
            for i, k in enumerate(ks):
                at_k = [r for r in got if r["k"] == k]
                for j, r in enumerate(at_k):
                    dx = (j - (len(at_k) - 1) / 2) * 7 + (-9 if condition == "top" else 9)
                    _mark(fig, which, panel.x(i) + dx, panel.y(r[field]), 3.8, f"f{colour} ring")
    bottom = top_px + 170 + 50
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def top_states(rows: list[dict], order: list[str]) -> tuple[list[int], list[dict]]:
    """Each K of top pruning, and each pole's state there against the instruct model.

    Args:
        rows: The paper's results.
        order: The poles, in the order of the grid.

    Returns:
        The values of K, and for each one ``{slug: state}``. A state is ``recovered``,
        ``failed``, ``lost`` or ``kept``.
    """
    sets = pole_sets(rows)
    none = sets["none", 0, None]
    ks = sorted(k for c, k, _ in sets if c == "top")
    states = []
    for k in ks:
        got = sets["top", k, None]
        states.append({s: ("recovered" if got[s] else "failed") if not none[s]
                       else ("lost" if not got[s] else "kept") for s in order})  # fmt: skip
    return ks, states


def grid(rows: list[dict]) -> Figure:
    """Each pole's state after top pruning, one frame per K.

    Args:
        rows: The paper's results.

    Returns:
        The figure. It steps through K and rests on the largest.
    """
    order = pole_order(rows, MODEL)
    ks, states = top_states(rows, order)
    fig = Figure(
        ["As more neurons go, top pruning breaks", "explanations at every margin"],
        [f"Each cell is one of Qwen2.5-0.5B's {len(order)} VERIFIED poles, nearest a gate first. "
         "Circles pass again after",
         "failing on the instruct model, triangles fail after keeping, dark cells still fail."],
        f"A grid of {len(order)} cells. The frames step through K = "
        f"{', '.join(map(str, ks))}. Lost poles grow from "
        f"{sum(v == 'lost' for v in states[0].values())} to "
        f"{sum(v == 'lost' for v in states[-1].values())}, and recovered poles stay between "
        f"{min(sum(v == 'recovered' for v in s.values()) for s in states)} and "
        f"{max(sum(v == 'recovered' for v in s.values()) for s in states)}.",
    )  # fmt: skip
    left, top = 40.0, fig.body_top + 34
    for i in range(len(order)):
        fig.bar(*_cell(i, left, top), CELL, CELL, "fwash")
    starts = [100 * i / len(ks) for i in range(len(ks))]
    for (cls, css), k, state in zip(counter("frame", starts, CYCLE), ks, states, strict=True):
        fig.motion += css
        lost = sum(v == "lost" for v in state.values())
        back = sum(v == "recovered" for v in state.values())
        fig.text(left, top - 12, f"K = {k}: {back} recovered, {lost} lost", f"head {cls}")
        for i, slug in enumerate(order):
            x, y = _cell(i, left, top)
            if state[slug] == "failed":
                fig.bar(x + 3, y + 3, CELL - 6, CELL - 6, f"fneutral {cls}")
            elif state[slug] != "kept":
                back = state[slug] == "recovered"
                which, colour = ("instruct", "c3") if back else ("random", "c2")
                _mark(fig, which, x + CELL / 2, y + CELL / 2, 5.2, f"f{colour} {cls}")
    bottom = _cell(len(order) - 1, left, top)[1] + CELL + GAP + 20
    fig.text(left, bottom, "Poles run left to right, then down: the nearest to a gate at the "
             "top left, the furthest at the bottom.", "small")  # fmt: skip
    fig.bottom = bottom
    return fig


# The paper's run and its variants (#7071): (unit, loss, label). Each has its own shape.
VARIANTS = (
    ("write", "base", "writes, corpus KL (the paper)"),
    ("input", "base", "input rows, corpus KL"),
    ("write", "failed", "writes, KL on half the failed texts"),
)
VARIANT_MARKS = (("instruct", "c1"), ("random", "c2"), ("square", "c3"))


def _variant_mark(fig: Figure, which: str, x: float, y: float, size: float, style: str) -> None:
    """One mark: a circle, a triangle or a square, so that no variant rests on colour alone.

    Args:
        fig: The figure.
        which: ``instruct`` (circle), ``random`` (triangle) or ``square``.
        x: The mark's centre, in pixels.
        y: The mark's centre, in pixels.
        size: The mark's half width, in pixels.
        style: The mark's CSS classes.
    """
    if which != "square":
        _mark(fig, which, x, y, size, style)
        return
    fig.path([(x - size, y - size), (x + size, y - size), (x + size, y + size),
              (x - size, y + size)], style, close=True)  # fmt: skip


def variants(rows: list[dict]) -> Figure:
    """The poles that each variant's top prune recovers and loses at each K.

    Args:
        rows: The paper's results, with the variants' rows.

    Returns:
        The figure.
    """
    rows = paper_poles_only(rows)
    tables = [[r for r in changes(rows, unit, loss) if r["condition"] == "top"]
              for unit, loss, _ in VARIANTS]  # fmt: skip
    ks = [r["k"] for r in tables[0]]
    failed, kept = tables[0][0]["failed_on_instruct"], tables[0][0]["kept_on_instruct"]
    lost_at = [t[-1]["lost"] for t in tables]
    fig = Figure(
        ["Neither variant recovers more explanations,", "and the failed-text loss loses more"],
        [f"The top K prune of each variant, on the paper's {failed + kept} poles. Left: of the "
         f"{failed} that fail on the instruct",
         f"model, those that pass again. Right: of the {kept} that keep on it, those that fail."],
        f"Two panels against K, from {ks[0]} to {ks[-1]}, one shape per variant. At K = "
        f"{ks[-1]} the paper's prune loses {lost_at[0]}, the input prune {lost_at[1]} and the "
        f"failed-text prune {lost_at[2]}.",
    )  # fmt: skip
    top_px = fig.body_top + 30
    for slot, (field, total, name) in enumerate((("recovered", failed, "recovered"),
                                                 ("lost", kept, "lost"))):  # fmt: skip
        span = 20 if field == "recovered" else 200
        panel = Panel(fig, (70 + slot * 350, top_px, 290, 170), (-0.5, len(ks) - 0.5), (0, span))
        panel.grid_y([0, span / 4, span / 2, 3 * span / 4, span], 0)
        panel.head(f"{name}, of {total}")
        for i, k in enumerate(ks):
            fig.text(panel.x(i), panel.top + panel.height + 15, str(k), "small", "middle")
        panel.x_title("K, the neurons pruned")
        for v, (table, (which, colour)) in enumerate(zip(tables, VARIANT_MARKS, strict=True)):
            for i, r in enumerate(table):
                x = panel.x(i) + (v - 1) * 9
                _variant_mark(fig, which, x, panel.y(r[field]), 3.8, f"f{colour} ring")
    y = top_px + 170 + 50
    for (_, _, label), (which, colour) in zip(VARIANTS, VARIANT_MARKS, strict=True):
        _variant_mark(fig, which, 25, y - 4, 5, f"f{colour}")
        fig.text(35, y, label)
        y += 18
    fig.bottom = y
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results, with the variants' rows.

    Returns:
        Each figure by its file name in ``media/``.
    """
    own = variant_rows(rows)
    return {"recovery.svg": recovery(own), "grid.svg": grid(own), "variants.svg": variants(rows)}
