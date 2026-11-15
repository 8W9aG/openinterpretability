"""The cross-model graph of sister neurons on a model's page.

There is one panel per other model that the sister search compared it with
(`open-interpretability sisters`, ``docs/analyses/sister-neurons.jsonl``). Each model is a column.
Its poles are placed by relative depth (layer over the deepest layer). So poles at the
same depth in two models of different depths sit level.

A line joins each VERIFIED pole to its best match, from either side's search. The higher
their correlation, the stronger the line. Sisters (mutual best matches above the rotated
basis) are drawn in colour, on top.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from openinterp.core.identifiers import model_slug
from openinterp.sisters import pole_href

_SLUG = re.compile(r"/L(\d+)/N(\d+)-(positive|negative)$")
WIDTH, HEIGHT, TOP, BOTTOM = 640, 380, 34, 14
COLUMNS = (190, 450)  # this model's column, then the other's
SPREAD = 60  # the most a layer's poles spread either side of their column


@dataclass
class _Edge:
    """A pole of this model (``a``), its match in the other (``b``), and how they match."""

    a: str
    b: str
    pearson: float
    sister: bool
    rotated: float


def _layer(slug: str) -> int:
    return int(_SLUG.search(slug).group(1))


def _edges(name: str, rows: dict) -> dict[str, dict[tuple[str, str], _Edge]]:
    """Return each other model's edges with ``name``.

    An edge is every row's best match, from either side, kept once per pair of poles. It
    is a sister if either side's search says so.
    """
    prefix = model_slug(name) + "/"
    names = _names(rows)
    out: dict[str, dict[tuple[str, str], _Edge]] = defaultdict(dict)
    for rs in rows.values():
        for row in rs:
            if row.top is None:
                continue
            match = row.top.slug(row.other)
            if row.slug.startswith(prefix):
                other, pair = row.other, (row.slug, match)
            elif row.other == name:
                other, pair = names[_source(row)], (match, row.slug)
            else:
                continue
            edge = _Edge(*pair, row.top.pearson, row.sister is not None, row.rotated)
            old = out[other].get(pair)
            if old is None or edge.sister or (edge.pearson > old.pearson and not old.sister):
                out[other][pair] = edge
    return out


def _source(row) -> str:
    """The slug of a row's source model."""
    return row.slug.split("/")[0]


def _names(rows: dict) -> dict[str, str]:
    """Return model names by slug, from the models that the rows were compared with.

    For a model that was only ever searched from, it undoes :func:`model_slug`.
    """
    known = {model_slug(r.other): r.other for rs in rows.values() for r in rs}
    sources = {_source(r) for rs in rows.values() for r in rs}
    return {s: known.get(s, s.replace("--", "/", 1)) for s in sources | set(known)}


def _depths(rows: dict) -> dict[str, int]:
    """Each model's layer count by name, as the rows record it."""
    names = _names(rows)
    out: dict[str, int] = {}
    for rs in rows.values():
        for row in rs:
            for key, n in ((names[_source(row)], row.layers), (row.other, row.other_layers)):
                if n:
                    out[key] = n
    return out


def _nodes(poles: set[str], x0: float, n_layers: int, studied: set) -> dict[str, dict]:
    """Node positions: y by relative depth, each layer's poles spread about ``x0``."""
    by_layer: dict[int, list[str]] = defaultdict(list)
    # The slug breaks ties between the two poles of a neuron. So the layout never depends on
    # the order of a set, which changes with Python's hash seed.
    for p in sorted(poles, key=lambda s: (*(int(g) for g in _SLUG.search(s).groups()[:2]), s)):
        by_layer[_layer(p)].append(p)
    out = {}
    for layer, ps in by_layer.items():
        step = min(5.0, 2 * SPREAD / max(len(ps) - 1, 1))
        for i, p in enumerate(ps):
            href = pole_href(p)
            out[p] = {
                "x": round(x0 + (i - (len(ps) - 1) / 2) * step, 1),
                "y": _y(layer, n_layers),
                "label": p.split("/", 1)[1].replace("-positive", "+").replace("-negative", "−"),
                "href": href if href in studied else None,
            }
    return out


def _y(layer: int, n_layers: int) -> float:
    return round(TOP + (HEIGHT - TOP - BOTTOM) * layer / max(n_layers - 1, 1), 1)


def _path(a: dict, b: dict) -> str:
    mid = (a["x"] + b["x"]) / 2
    return f"M{a['x']} {a['y']} C{mid} {a['y']} {mid} {b['y']} {b['x']} {b['y']}"


def _ticks(x: float, n_layers: int, anchor: str) -> list[dict]:
    every = 1 if n_layers <= 12 else 2 if n_layers <= 24 else 3
    layers = sorted({*range(0, n_layers, every), n_layers - 1})
    return [{"x": x, "y": _y(i, n_layers), "label": f"L{i}", "anchor": anchor} for i in layers]


def sister_graphs(model, rows: dict, studied: set) -> list[dict]:
    """The model page's panels, one per other model, most sisters first."""
    depths = _depths(rows)
    panels = []
    for other, edges in _edges(model.name, rows).items():
        n_a = depths.get(model.name) or model.n_layers
        n_b = depths.get(other) or 1 + max(_layer(e.b) for e in edges.values())
        panels.append(_panel(model, other, list(edges.values()), (n_a, n_b), studied))
    return sorted(panels, key=lambda p: (-p["n_sisters"], p["other"]))


def _panel(model, other: str, edges: list[_Edge], depth: tuple[int, int], studied) -> dict:
    """One other model's panel: nodes, lines (weakest first, sisters last) and sisters."""
    (xa, xb), (n_a, n_b) = COLUMNS, depth
    nodes = {
        **_nodes({e.a for e in edges}, xa, n_a, studied),
        **_nodes({e.b for e in edges}, xb, n_b, studied),
    }
    edges.sort(key=lambda e: (e.sister, e.pearson))
    sisters = [e for e in edges if e.sister]
    lines = [
        {
            "d": _path(nodes[e.a], nodes[e.b]),
            "sister": e.sister,
            "opacity": round(min(max(e.pearson, 0.05), 1.0), 2),
            "label": f"{nodes[e.a]['label']} ↔ {nodes[e.b]['label']}: r {e.pearson:.2f}",
        }
        for e in edges
    ]
    gaps = [abs(_layer(e.a) / max(n_a - 1, 1) - _layer(e.b) / max(n_b - 1, 1)) for e in sisters]
    return {
        "other": other,
        "width": WIDTH,
        "height": HEIGHT,
        "columns": COLUMNS,
        "nodes": list(nodes.values()),
        "lines": lines,
        "ticks": _ticks(xa - SPREAD - 14, n_a, "end") + _ticks(xb + SPREAD + 14, n_b, "start"),
        "n_edges": len(edges),
        "n_sisters": len(sisters),
        "depth_gap": round(median(gaps), 3) if gaps else None,
        "sisters": [_sister_line(e, nodes) for e in reversed(sisters)],
    }


def _sister_line(e: _Edge, nodes: dict) -> dict:
    return {"a": nodes[e.a], "b": nodes[e.b], "pearson": e.pearson, "rotated": e.rotated}
