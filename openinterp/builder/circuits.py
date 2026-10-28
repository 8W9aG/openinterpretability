"""Circuit pages: the registry's units joined by their wiring edges.

Every experiment's wiring (its own, or a backfilled row) names its unit's strongest
direct connections. An edge between two units that both have pages is kept; the units
it joins, followed edge to edge, form one circuit per connected group. Each circuit
is laid out by layer, so chains read left to right from earlier to later layers.
:func:`wiring_view` is one experiment's wiring, with each edge's page linked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from openinterp.builder.tree import ModelNode, UnitNode, head_href, neuron_href
from openinterp.wiring_backfill import record_wiring

HREF = {"mlp_neuron": neuron_href, "attn_head": head_href}
COLUMN, ROW, PAD = 120, 44, 40


@dataclass
class Reading:
    """One reading of one edge: which experiment's wiring found it, and how."""

    unit: UnitNode  # whose experiment it is
    experiment: object  # builder.tree.Experiment
    method: str
    weight: float
    channels: dict[str, float]
    backfilled: bool


@dataclass
class CircuitEdge:
    source: UnitNode
    target: UnitNode
    readings: list[Reading] = field(default_factory=list)

    @property
    def methods(self) -> list[str]:
        return sorted({r.method for r in self.readings})


@dataclass
class Circuit:
    units: list[UnitNode]
    edges: list[CircuitEdge]
    positions: dict[str, tuple[int, int]]  # href -> (x, y) of the node's centre
    width: int
    height: int

    @property
    def layers(self) -> tuple[int, int]:
        return min(u.layer for u in self.units), max(u.layer for u in self.units)


def circuit_edges(model: ModelNode, rows: dict) -> list[CircuitEdge]:
    """Edges between two of ``model``'s units with pages, from every experiment's wiring,
    each found once per reading that names it."""
    units = {u.href: u for layer in model.layers for u in layer.units}
    edges: dict[tuple[str, str], CircuitEdge] = {}
    for unit in units.values():
        for e in unit.experiments:
            weights, activations, backfilled = record_wiring(e.record, rows)
            for wiring in (weights, activations):
                if wiring is None:
                    continue
                sides = [
                    (wiring.upstream_neurons, True),
                    (wiring.upstream_heads or [], True),
                    (wiring.downstream_neurons, False),
                    (wiring.downstream_heads or [], False),
                ]
                for found, upstream in sides:
                    for edge in found:
                        other = units.get(HREF[edge.kind](model.slug, edge.layer, edge.index))
                        if other is None or other is unit:
                            continue
                        source, target = (other, unit) if upstream else (unit, other)
                        key = (source.href, target.href)
                        if key not in edges:
                            edges[key] = CircuitEdge(source, target)
                        edges[key].readings.append(
                            Reading(unit, e, wiring.method, edge.weight, edge.channels, backfilled)
                        )
    return sorted(
        edges.values(), key=lambda c: (c.source.layer, c.target.layer, c.source.href, c.target.href)
    )


def circuits(model: ModelNode, rows: dict) -> list[Circuit]:
    """``model``'s circuits, largest first: groups of units joined by wiring edges."""
    edges = circuit_edges(model, rows)
    parent: dict[str, str] = {}

    def find(h: str) -> str:
        while parent.setdefault(h, h) != h:
            parent[h] = parent[parent[h]]
            h = parent[h]
        return h

    for edge in edges:
        parent[find(edge.source.href)] = find(edge.target.href)
    groups: dict[str, list[CircuitEdge]] = defaultdict(list)
    for edge in edges:
        groups[find(edge.source.href)].append(edge)
    found = [_layout(group) for group in groups.values()]
    return sorted(found, key=lambda c: (-len(c.units), c.layers, c.units[0].href))


def _layout(edges: list[CircuitEdge]) -> Circuit:
    """Columns by layer (heads before neurons, as attention runs first), rows within."""
    units = {u.href: u for e in edges for u in (e.source, e.target)}
    ordered = sorted(units.values(), key=lambda u: (u.layer, not u.is_head, u.idx))
    columns: dict[int, list[UnitNode]] = defaultdict(list)
    for u in ordered:
        columns[u.layer].append(u)
    positions = {}
    for col, layer in enumerate(sorted(columns)):
        for row, u in enumerate(columns[layer]):
            positions[u.href] = (PAD + col * COLUMN, PAD + row * ROW)
    width = 2 * PAD + (len(columns) - 1) * COLUMN
    height = 2 * PAD + (max(len(c) for c in columns.values()) - 1) * ROW
    return Circuit(ordered, edges, positions, width, height)


def unit_circuits(model: ModelNode, found: list[Circuit]) -> dict[str, int]:
    """Each unit's circuit number (1-based, as on the page), by href."""
    return {u.href: n for n, c in enumerate(found, 1) for u in c.units}


def wiring_view(wiring, slug: str, studied: set[str]) -> dict | None:
    """A wiring's edge lists with each edge's page linked when the registry has one."""
    if wiring is None:
        return None

    def rows(edges):
        if edges is None:
            return None
        out = []
        for edge in edges:
            page = HREF[edge.kind](slug, edge.layer, edge.index)
            # The channel the weight came from, when there are several to choose from.
            via = max(edge.channels, key=lambda c: abs(edge.channels[c]), default=None)
            out.append(
                {
                    "e": edge,
                    "href": page if page in studied else None,
                    "via": via if len(edge.channels) > 1 else None,
                }
            )
        return out

    return {
        "w": wiring,
        "groups": [
            ("Upstream neurons", rows(wiring.upstream_neurons)),
            ("Upstream heads", rows(wiring.upstream_heads)),
            ("Downstream neurons", rows(wiring.downstream_neurons)),
            ("Downstream heads", rows(wiring.downstream_heads)),
        ],
    }
