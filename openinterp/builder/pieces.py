"""Pieces bars: which neurons of each layer are studied, and their status.

The home page draws one bar per layer of a model (:func:`model_pieces`), and a layer page
draws one cell per neuron (:func:`layer_pieces`). A neuron whose scans all stopped is
studied too (:class:`openinterp.builder.stopped.StoppedNeuron`).
"""

from __future__ import annotations

import math

from openinterp.builder.charts import best_status, pole_text, stopped_label
from openinterp.builder.statuses import MAP_STATUSES, STOPPED
from openinterp.builder.stopped import StoppedNeuron

# Pieces bars: BitTorrent-style coverage, one cell per neuron or block of neurons. Cells sit
# on an integer pitch so the empty-cell grid can be one SVG pattern rather than a rect each.
PIECE_ROW = 128  # cells per bar
PIECE_PITCH = 5  # px per cell, 1px gap included


def model_pieces(model) -> dict | None:
    """One pieces bar per layer, each cell a block of ``ceil(mlp_width / PIECE_ROW)`` neurons.

    A cell is lit when any neuron in its block is studied, and takes the best status among
    them (``MAP_STATUSES`` order). With blocks this wide a lit cell is not a covered block:
    the bars show where the studied neurons are and where the gaps are, not how many.
    """
    if not model.mlp_width or not model.neurons:
        return None
    block = math.ceil(model.mlp_width / PIECE_ROW)
    n_cells = math.ceil(model.mlp_width / block)
    by_layer = {layer.layer: layer for layer in model.layers}
    rows = []
    for i in range(model.n_layers):
        cells: dict[int, list] = {}
        layer = by_layer.get(i)
        for n in [*layer.neurons, *layer.stopped] if layer else ():
            cells.setdefault(n.idx // block, []).append(n)
        lit = []
        for c, neurons in sorted(cells.items()):
            status = min((cell_status(n) for n in neurons), key=MAP_STATUSES.index)
            lo, hi = c * block, min((c + 1) * block, model.mlp_width) - 1
            lit.append(
                {
                    "x": c * PIECE_PITCH,
                    "status": status,
                    "label": f"L{i} N{lo:,}–{hi:,}: " + "; ".join(map(cell_label, neurons)),
                }
            )
        rows.append({"layer": i, "cells": lit})
    return {
        "block": block,
        "pitch": PIECE_PITCH,
        "width": n_cells * PIECE_PITCH - 1,
        "rows": rows,
        "lit": sum(len(r["cells"]) for r in rows),
        "cells": n_cells * model.n_layers,
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

    def place(idx: int) -> dict:
        return {
            "x": left + (idx % PIECE_ROW) * PIECE_PITCH,
            "y": top + (idx // PIECE_ROW) * PIECE_PITCH,
        }

    studied = [
        place(n.idx)
        | {"status": best_status(n), "href": n.href, "label": f"{n.label} · {pole_text(n)}"}
        for n in layer.neurons
    ]
    stopped = [
        place(n.idx) | {"status": STOPPED, "href": None, "label": stopped_label(n)}
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
