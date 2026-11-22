"""Status grids: which neurons of a model or a layer are studied, and their best status.

One function, :func:`status_grid`, gives the geometry of all three charts (#5797). The
``status_grid`` macro in ``_macros.html`` draws each one, with its legend.

- **A model card** (home and Models pages) and **a model page** show one row per layer. Each
  cell is a block of neurons, so that a layer has at most :data:`GRID_ROW` cells.
- **A layer page** shows one cell per neuron, :data:`GRID_ROW` to a row.

On a model's grid, a cell is a stacked bar of counts (#5836). Its height is the number of
studied neurons in the block, :data:`~.statuses.NEURON_PX` each, on one scale for every
model (:func:`grid_scale`). Its segments are those neurons by status, in ``MAP_STATUSES``
order from the bottom, so VERIFIED is always at the bottom. A gap separates two statuses,
so a colour-blind reader can tell them apart by place. On a layer page, a cell is one
neuron, full height in its status's colour. No two cells overlap. A neuron whose scans all
stopped is studied too (:class:`openinterpretability.builder.stopped.StoppedNeuron`).
"""

from __future__ import annotations

import math

from openinterpretability.builder.charts import best_status, pole_text, stopped_label
from openinterpretability.builder.statuses import (
    CELL_HEIGHT,
    MAP_STATUSES,
    NEURON_PX,
    SEGMENT_GAP,
    STOPPED,
)
from openinterpretability.builder.stopped import StoppedNeuron

GRID_ROW = 128  # cells per row
GRID_PITCH = 7  # px per cell across, 1 px gap included
ROW_GAP = 2  # px between two rows
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


def _blocks(node, block: int) -> dict[int, list]:
    """The studied neurons of a layer, by the block that holds them.

    Args:
        node: The layer's node.
        block: The neurons per cell.

    Returns:
        Each block's neurons, by the block's number in the layer.
    """
    blocks: dict[int, list] = {}
    for n in [*node.neurons, *node.stopped]:
        blocks.setdefault(n.idx // block, []).append(n)
    return blocks


def grid_scale(models) -> int:
    """The most studied neurons in any block of any model's grid: one scale for all (#5836).

    Every model's grid makes its rows this many neurons high, so the tallest stack fits its
    row and two cards compare.

    Args:
        models: The models' nodes.

    Returns:
        The largest count, at least 1.

    >>> grid_scale([])
    1
    """
    counts = [len(neurons) for m in models if m.mlp_width
              for node in m.layers for neurons in _blocks(node, _block(m)).values()]  # fmt: skip
    return max(counts, default=1)


def _block(model) -> int:
    """The neurons per cell of a model's grid, so that a layer fits one row.

    Args:
        model: The model's node.

    Returns:
        The block size.
    """
    return math.ceil(model.mlp_width / GRID_ROW)


def stack_height(count: int) -> int:
    """The most height that ``count`` studied neurons take, gaps included.

    Args:
        count: Studied neurons in one block.

    Returns:
        Its height in px, as if each neuron had its own status.

    >>> stack_height(1), stack_height(4)
    (3, 15)
    """
    return count * (NEURON_PX + SEGMENT_GAP) - SEGMENT_GAP


def status_grid(model, layer=None, scale: int | None = None) -> dict | None:
    """Return the cells, rows and legend of a model's or a layer's status grid (#5797).

    Args:
        model: The model's node in the site tree.
        layer: A layer's node, for a layer page's grid of one cell per neuron. None for a
            model's grid of one row per layer.
        scale: The most studied neurons in any block, for the rows' height (:func:`grid_scale`).
            None for this model's own most.

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
    block = 1 if layer else _block(model)
    tall = CELL_HEIGHT if layer else stack_height(scale or grid_scale([model]))
    per_layer = math.ceil(model.mlp_width / block)
    by_layer = {node.layer: node for node in model.layers}
    layers = [layer] if layer else [by_layer.get(i) for i in range(model.n_layers)]
    rows, cells, counts = [], [], dict.fromkeys(MAP_STATUSES, 0)
    for i, node in enumerate(layers):
        first = len(rows)
        for r in range(math.ceil(per_layer / GRID_ROW)):
            rows.append(_row(len(rows), r * GRID_ROW, block, per_layer, layer, i, tall))
        # The legend counts neurons, not cells: a cell can hold neurons of several statuses.
        for neuron in [*node.neurons, *node.stopped] if node else ():
            counts[cell_status(neuron)] += 1
        cells += _cells(node, block, first, model.mlp_width, tall) if node else []
    studied = sum(counts.values())
    return {
        "block": block,
        "per_row": min(GRID_ROW, per_layer),
        "left": LABEL_WIDTH,
        "pitch": GRID_PITCH,
        "cell": tall,
        "width": LABEL_WIDTH + GRID_ROW * GRID_PITCH - 1,
        "height": len(rows) * (tall + ROW_GAP) - ROW_GAP,
        "rows": rows,
        "cells": cells,
        "legend": [(s, counts[s]) for s in MAP_STATUSES],
        "studied": studied,
        "unexplored": model.mlp_width * len(layers) - studied,
    }


def _row(r: int, start: int, block: int, per_layer: int, layer, i: int, tall: int) -> dict:
    """Return one row of the grid: its place, the width of its cells and its label.

    Args:
        r: The row's place in the grid, from the top.
        start: The row's first cell, in its layer.
        block: The neurons per cell.
        per_layer: The cells per layer.
        layer: The layer's node on a layer page, else None.
        i: The row's layer, on a model's grid.
        tall: The row's height in px.

    Returns:
        The row's ``y``, ``width``, ``label`` and whether to show the label.
    """
    shown = (r % LAYER_LABEL_EVERY == 0) if layer else True
    return {
        "y": r * (tall + ROW_GAP),
        "width": min(GRID_ROW, per_layer - start) * GRID_PITCH - 1,
        "label": f"N{start * block:,}" if layer else f"L{i}",
        "shown": shown,
    }


def _cells(node, block: int, first: int, width: int, tall: int) -> list[dict]:
    """Return one cell per block of ``node``'s neurons that holds a studied neuron.

    Args:
        node: The layer's node.
        block: The neurons per cell.
        first: The grid row of the layer's first cell.
        width: The layer's MLP width.
        tall: The row's height in px.

    Returns:
        Each cell's place, height, best status, segments, hover text and link.
    """
    cells = []
    for c, neurons in sorted(_blocks(node, block).items()):
        row, col = first + c // GRID_ROW, c % GRID_ROW
        bottom = row * (tall + ROW_GAP) + tall
        segments = _segments(neurons, bottom) if block > 1 else [
            {"y": bottom - tall, "height": tall, "status": cell_status(neurons[0])}]  # fmt: skip
        label, href = _cell_text(node, neurons, c, block, width)
        top = min(s["y"] for s in segments)
        cells.append({
            "x": LABEL_WIDTH + col * GRID_PITCH,
            "y": top,
            "height": bottom - top,
            "status": segments[0]["status"],  # the best status, at the bottom
            "segments": segments,
            "label": label,
            "href": href,
        })  # fmt: skip
    return cells


def _segments(neurons: list, bottom: int) -> list[dict]:
    """A block's stack: one segment per status, in ``MAP_STATUSES`` order from ``bottom`` up.

    Args:
        neurons: The block's studied neurons.
        bottom: The row's bottom edge, in px.

    Returns:
        Each segment's top, height (its count × NEURON_PX) and status.

    >>> from openinterpretability.builder.stopped import StoppedNeuron
    >>> _segments([StoppedNeuron(1, 2, ["positive dead"])] * 2, 20)
    [{'y': 14, 'height': 6, 'status': 'STOPPED'}]
    """
    by_status = {s: sum(cell_status(n) == s for n in neurons) for s in MAP_STATUSES}
    out, y = [], bottom
    for status, count in by_status.items():
        if count:
            height = count * NEURON_PX
            out.append({"y": y - height, "height": height, "status": status})
            y -= height + SEGMENT_GAP
    return out


def _cell_text(node, neurons: list, c: int, block: int, width: int) -> tuple[str, str]:
    """A cell's hover text and link: the neuron's page, or for a block its layer's page.

    A block's text gives its counts by status, then each neuron (#5836).

    Args:
        node: The layer's node.
        neurons: The cell's studied neurons.
        c: The cell's number in the layer.
        block: The neurons per cell.
        width: The layer's MLP width.

    Returns:
        ``(label, href)``.
    """
    poles = sum(map(studied_poles, neurons))
    if block == 1:
        (neuron,) = neurons
        return f"{cell_label(neuron)}, {poles} of 2 poles studied", neuron.href
    lo, hi = c * block, min((c + 1) * block, width) - 1
    found = [(s, sum(cell_status(n) == s for n in neurons)) for s in MAP_STATUSES]
    counts = ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in found if n)
    label = f"L{node.layer} N{lo:,}–{hi:,}: {len(neurons)} studied: {counts}. "
    label += f"{poles} of {2 * (hi - lo + 1)} poles studied: " + "; ".join(map(cell_label, neurons))
    return label, node.href
