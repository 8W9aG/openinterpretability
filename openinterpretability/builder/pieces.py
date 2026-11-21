"""Pieces bars: which neurons of each layer are studied, and their status.

The home and Models pages draw one bar per layer of a model (:func:`model_pieces`). A
layer page draws one cell per neuron (:func:`layer_pieces`). A neuron whose scans all
stopped is studied too (:class:`openinterpretability.builder.stopped.StoppedNeuron`).

On a layer map, a cell's colour is the best status in it. How far it fills, from the
bottom, is the share of its poles that are studied, two per neuron (#3288). Length carries
the amount, not opacity, because faded status colours can't be told apart by colour-blind
readers. On a model card, a cell's height says whether its best status is VERIFIED (#5724).
"""

from __future__ import annotations

import math

from openinterpretability.builder.charts import best_status, pole_text, stopped_label
from openinterpretability.builder.statuses import MAP_STATUSES, STOPPED
from openinterpretability.builder.stopped import StoppedNeuron

# Pieces bars: BitTorrent-style coverage, one cell per neuron or block of neurons. Cells sit
# on an integer pitch so the empty-cell grid can be one SVG pattern rather than a rect each.
PIECE_ROW = 128  # cells per row of a layer map
PIECE_PITCH = 5  # px per cell, 1px gap included; a layer map's cell is PIECE_PITCH - 1
# A model card's bars (#5724). Every model has the same cells per row and the same row
# height, so one cell is one 64th of a layer on every card. The bars stretch to the card's
# width, and each row stays CARD_ROW_PX high on every screen.
CARD_ROW = 64  # cells per layer
CARD_PITCH = 8  # viewBox units per cell, 1 unit gap included: a thin gap once stretched
CARD_ROW_PX = 6  # px per layer, 1px gap included
CARD_CELL = 5  # px of a cell with a VERIFIED pole
CARD_SHORT = 2  # px of a studied cell with no VERIFIED pole


def studied_poles(neuron) -> int:
    """How many of a studied neuron's two poles are studied: with experiments, or stopped."""
    if isinstance(neuron, StoppedNeuron):
        return len({outcome.split()[0] for outcome in neuron.outcomes}) or 1
    return len(neuron.standings) or 1


def fill(share: float, height: int) -> int:
    """The filled height of a cell: its share of ``height``, at least 1 px when any is studied."""
    return max(1, min(height, round(share * height))) if share > 0 else 0


def model_pieces(model) -> dict | None:
    """One row of ``CARD_ROW`` cells per layer, each cell a block of neurons (#5724).

    A cell is lit when any neuron in its block is studied, and takes the best status among
    them (``MAP_STATUSES`` order). Its height says, without colour, whether that status is
    VERIFIED: a VERIFIED cell is ``CARD_CELL`` px high, and any other is ``CARD_SHORT``.
    The hover text gives the exact counts.

    Args:
        model: The model's node in the site tree.

    Returns:
        The geometry and cells of the bars, or None for a model with no MLP width or no
        studied neuron.
    """
    if not model.mlp_width or not model.neurons:
        return None
    # The block size differs by model, so that every model has CARD_ROW cells per layer.
    block = math.ceil(model.mlp_width / CARD_ROW)
    n_cells = math.ceil(model.mlp_width / block)
    by_layer = {layer.layer: layer for layer in model.layers}
    rows = []
    for i in range(model.n_layers):  # every layer has a row, studied or not
        cells: dict[int, list] = {}
        layer = by_layer.get(i)
        for n in [*layer.neurons, *layer.stopped] if layer else ():
            cells.setdefault(n.idx // block, []).append(n)
        lit = [_card_cell(i, c, cells[c], block, model.mlp_width) for c in sorted(cells)]
        rows.append({"layer": i, "cells": lit})
    return {
        "block": block,
        "pitch": CARD_PITCH,
        "row": CARD_ROW_PX,
        "cell": CARD_CELL,
        "short": CARD_SHORT,
        "width": n_cells * CARD_PITCH - 1,
        "height": model.n_layers * CARD_ROW_PX - 1,
        "rows": rows,
        "lit": sum(len(r["cells"]) for r in rows),
        "verified": sum(c["status"] == "VERIFIED" for r in rows for c in r["cells"]),
        "cells": n_cells * model.n_layers,
    }


def _card_cell(layer: int, c: int, neurons: list, block: int, width: int) -> dict:
    """Return one lit cell of a model card's bars (#5724).

    Args:
        layer: The cell's layer, which is its row.
        c: The cell's place in the row.
        neurons: The studied neurons in the cell's block.
        block: The neurons per cell.
        width: The layer's MLP width.

    Returns:
        The cell's place, height, best status and hover text.
    """
    status = min((cell_status(n) for n in neurons), key=MAP_STATUSES.index)
    lo, hi = c * block, min((c + 1) * block, width) - 1
    poles, of = sum(map(studied_poles, neurons)), 2 * (hi - lo + 1)
    height = CARD_CELL if status == "VERIFIED" else CARD_SHORT
    return {
        "x": c * CARD_PITCH,
        "y": layer * CARD_ROW_PX + CARD_CELL - height,  # from the bottom of the row
        "height": height,
        "status": status,
        "label": f"L{layer} N{lo:,}–{hi:,}, {poles} of {of} poles studied: "
        + "; ".join(map(cell_label, neurons)),
    }


def cell_status(neuron) -> str:
    """A studied neuron's map status: its best pole's, or STOPPED."""
    return STOPPED if isinstance(neuron, StoppedNeuron) else best_status(neuron)


def cell_label(neuron) -> str:
    """A studied neuron's text in a pieces cell."""
    if isinstance(neuron, StoppedNeuron):
        return stopped_label(neuron)
    return f"{neuron.label} {pole_text(neuron)}"


def _layer_cells(layer, left: int, top: int) -> list[dict]:
    """Return one cell per studied neuron of ``layer``, at its place in the grid."""
    side = PIECE_PITCH - 1

    def place(neuron) -> dict:
        """The cell's box: filled from the bottom by the share of its two poles studied."""
        height = fill(studied_poles(neuron) / 2, side)
        return {
            "x": left + (neuron.idx % PIECE_ROW) * PIECE_PITCH,
            "y": top + (neuron.idx // PIECE_ROW) * PIECE_PITCH + side - height,
            "height": height,
        }

    studied = [
        place(n)
        | {"status": best_status(n), "href": n.href, "label": f"{n.label} · {pole_text(n)}"}
        for n in layer.neurons
    ]
    stopped = [
        place(n) | {"status": STOPPED, "href": n.href, "label": stopped_label(n)}
        for n in layer.stopped
    ]
    return studied + stopped


def layer_pieces(model, layer) -> dict | None:
    """Return the layer's neurons as a grid of ``PIECE_ROW``-wide rows, one cell per neuron.

    Each studied neuron is a lit cell, coloured by its best pole's status, linked to its page.
    """
    if not model.mlp_width:
        return None
    left, top = 8 * PIECE_PITCH, 0  # room for row labels; multiples of the pitch keep
    # the empty-cell pattern aligned with the lit cells
    n_rows = math.ceil(model.mlp_width / PIECE_ROW)
    cells = _layer_cells(layer, left, top)
    rows = [
        {
            "y": top + r * PIECE_PITCH,
            "start": r * PIECE_ROW,
            "width": min(PIECE_ROW, model.mlp_width - r * PIECE_ROW) * PIECE_PITCH - 1,
        }
        for r in range(n_rows)
    ]
    counts = dict.fromkeys(MAP_STATUSES, 0)
    for c in cells:
        counts[c["status"]] += 1
    return {
        "left": left,
        "pitch": PIECE_PITCH,
        "width": left + PIECE_ROW * PIECE_PITCH,
        "height": top + n_rows * PIECE_PITCH,
        "rows": rows,
        "label_every": 8,
        "cells": cells,
        "counts": counts,
        "unexplored": model.mlp_width - len(cells),
    }
