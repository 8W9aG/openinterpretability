"""Status grids: which neurons of a model or a layer are studied, and their best status.

One function, :func:`status_grid`, gives the geometry of all three charts (#5797). The
``status_grid`` macro in ``_macros.html`` draws each one, with its legend.

- **A model card** (home and Models pages) and **a model page** show one row per layer. Each
  cell is a block of neurons, so that a layer has at most :data:`GRID_ROW` cells.
- **A layer page** shows one cell per neuron, :data:`GRID_ROW` to a row.

A cell's colour and its height both give the best status in it (``STATUS_HEIGHT``), so a
colour-blind reader can read it. No two cells overlap. A neuron whose scans all stopped is
studied too (:class:`openinterpretability.builder.stopped.StoppedNeuron`).
"""

from __future__ import annotations

import math

from openinterpretability.builder.charts import best_status, pole_text, stopped_label
from openinterpretability.builder.statuses import (
    CELL_HEIGHT,
    MAP_STATUSES,
    STATUS_HEIGHT,
    STOPPED,
)
from openinterpretability.builder.stopped import StoppedNeuron

GRID_ROW = 128  # cells per row
GRID_PITCH = 7  # px per cell across, 1 px gap included
ROW_PITCH = CELL_HEIGHT + 2  # px per row, 2 px gap included
LABEL_WIDTH = 56  # px for the row labels, "L23" or "N8,832", left of the cells
LAYER_LABEL_EVERY = 4  # a layer page labels every fourth row of neurons


def studied_poles(neuron) -> int:
    """How many of a studied neuron's two poles are studied: with experiments, or stopped."""
    if isinstance(neuron, StoppedNeuron):
        return len({outcome.split()[0] for outcome in neuron.outcomes}) or 1
    return len(neuron.standings) or 1


def cell_status(neuron) -> str:
    """A studied neuron's map status: its best pole's, or STOPPED."""
    return STOPPED if isinstance(neuron, StoppedNeuron) else best_status(neuron)


def cell_label(neuron) -> str:
    """A studied neuron's text in a grid cell."""
    if isinstance(neuron, StoppedNeuron):
        return stopped_label(neuron)
    return f"{neuron.label} {pole_text(neuron)}"


def status_grid(model, layer=None) -> dict | None:
    """Return the cells, rows and legend of a model's or a layer's status grid (#5797).

    Args:
        model: The model's node in the site tree.
        layer: A layer's node, for a layer page's grid of one cell per neuron. None for a
            model's grid of one row per layer.

    Returns:
        The geometry, or None for a model with no MLP width, or with no studied neuron.

    >>> from openinterpretability.builder.tree import ModelNode
    >>> status_grid(ModelNode("m", "m", 2, "m/index.html")) is None  # no MLP width
    True
    """
    if not model.mlp_width or (layer is None and not model.neurons):
        return None
    # On a model's grid, a cell is a block of neurons. So a layer fits one row, and no two
    # cells overlap. A layer's grid has one cell per neuron, and its rows wrap.
    block = 1 if layer else math.ceil(model.mlp_width / GRID_ROW)
    per_layer = math.ceil(model.mlp_width / block)
    by_layer = {node.layer: node for node in model.layers}
    layers = [layer] if layer else [by_layer.get(i) for i in range(model.n_layers)]
    rows, cells, counts = [], [], dict.fromkeys(MAP_STATUSES, 0)
    for i, node in enumerate(layers):
        first = len(rows)
        for r in range(math.ceil(per_layer / GRID_ROW)):
            start = r * GRID_ROW
            rows.append(_row(len(rows), start, block, per_layer, layer, i))
        # The legend counts neurons, not cells: a cell can hold neurons of several statuses.
        for neuron in [*node.neurons, *node.stopped] if node else ():
            counts[cell_status(neuron)] += 1
        cells += _cells(node, block, first, model.mlp_width) if node else []
    studied = sum(counts.values())
    return {
        "block": block,
        "per_row": min(GRID_ROW, per_layer),
        "left": LABEL_WIDTH,
        "pitch": GRID_PITCH,
        "cell": CELL_HEIGHT,
        "width": LABEL_WIDTH + GRID_ROW * GRID_PITCH - 1,
        "height": len(rows) * ROW_PITCH - 2,
        "rows": rows,
        "cells": cells,
        "legend": [(s, STATUS_HEIGHT[s], counts[s]) for s in MAP_STATUSES],
        "studied": studied,
        "unexplored": model.mlp_width * len(layers) - studied,
    }


def _row(r: int, start: int, block: int, per_layer: int, layer, i: int) -> dict:
    """Return one row of the grid: its place, the width of its cells and its label.

    Args:
        r: The row's place in the grid, from the top.
        start: The row's first cell, in its layer.
        block: The neurons per cell.
        per_layer: The cells per layer.
        layer: The layer's node on a layer page, else None.
        i: The row's layer, on a model's grid.

    Returns:
        The row's ``y``, ``width``, ``label`` and whether to show the label.
    """
    shown = (r % LAYER_LABEL_EVERY == 0) if layer else True
    return {
        "y": r * ROW_PITCH,
        "width": min(GRID_ROW, per_layer - start) * GRID_PITCH - 1,
        "label": f"N{start * block:,}" if layer else f"L{i}",
        "shown": shown,
    }


def _cells(node, block: int, first: int, width: int) -> list[dict]:
    """Return one cell per block of ``node``'s neurons that holds a studied neuron.

    Args:
        node: The layer's node.
        block: The neurons per cell.
        first: The grid row of the layer's first cell.
        width: The layer's MLP width.

    Returns:
        Each cell's place, height, best status, hover text and link.
    """
    blocks: dict[int, list] = {}
    for n in [*node.neurons, *node.stopped]:
        blocks.setdefault(n.idx // block, []).append(n)
    cells = []
    for c, neurons in sorted(blocks.items()):
        status = min(map(cell_status, neurons), key=MAP_STATUSES.index)
        height = STATUS_HEIGHT[status]
        row, col = first + c // GRID_ROW, c % GRID_ROW
        lo, hi = c * block, min((c + 1) * block, width) - 1
        poles = sum(map(studied_poles, neurons))
        if block == 1:
            (neuron,) = neurons
            label, href = f"{cell_label(neuron)}, {poles} of 2 poles studied", neuron.href
        else:
            label = f"L{node.layer} N{lo:,}–{hi:,}, {poles} of {2 * (hi - lo + 1)} poles "
            label += "studied: " + "; ".join(map(cell_label, neurons))
            href = node.href
        cells.append({
            "x": LABEL_WIDTH + col * GRID_PITCH,
            "y": row * ROW_PITCH + CELL_HEIGHT - height,  # from the bottom of the row
            "height": height,
            "status": status,
            "label": label,
            "href": href,
        })  # fmt: skip
    return cells
