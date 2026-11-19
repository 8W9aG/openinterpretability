"""Circuit pages: the registry's units joined by their wiring edges.

Every experiment's wiring (its own, or a backfilled row) names its unit's strongest
direct connections. An edge between two units that both have pages is kept. The units
that edges join form one circuit per connected group. Each circuit
is laid out by layer, so chains read left to right from earlier to later layers.

A *hub* is a unit wired to a large share of the model's studied units. An example is a
late neuron that every earlier unit's write moves a little. A hub would join them all
into one star-shaped circuit. So a hub's edges don't join circuits, and the page lists
hubs separately.

Each circuit and hub has its own page, named after one unit (:func:`unit_key`). A circuit
is named after its *root*, its first unit in layout order: the lowest layer, heads
first, then the lowest index. Circuits merge and split as the registry grows, so a root
can change. :mod:`openinterpretability.builder.circuit_pages` therefore writes a redirect at every
other unit's name. A URL from an earlier build then lands on the unit's current page.

:func:`wiring_view` is one experiment's wiring, with each edge's page linked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

from openinterpretability.builder.hrefs import circuit_href, hub_href
from openinterpretability.builder.tree import ModelNode, UnitNode, head_href, neuron_href
from openinterpretability.wiring_backfill import record_wiring

HREF = {"mlp_neuron": neuron_href, "attn_head": head_href}
COLUMN, ROW, PAD = 120, 44, 40
# A hub is wired to at least this share of the model's units with pages, and this many.
HUB_SHARE, HUB_MIN = 0.05, 10
HUB_RULE = {"hub_share": HUB_SHARE, "hub_min": HUB_MIN}  # for the page


def unit_key(unit: UnitNode) -> str:
    """The unit's name in circuit and hub URLs, from its page: ``L3-N7``, ``L4-tres-dc-F12``."""
    within = unit.href.removesuffix(".html").split("/", 2)[2]  # after models/<slug>/
    return within.replace("/", "-")


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
    """One connection between two units, with every reading that names it."""

    source: UnitNode
    target: UnitNode
    readings: list[Reading] = field(default_factory=list)

    @property
    def methods(self) -> list[str]:
        """The wiring methods that found this edge, sorted."""
        return sorted({r.method for r in self.readings})


@dataclass
class Circuit:
    """One connected group of units, laid out by layer."""

    units: list[UnitNode]
    edges: list[CircuitEdge]
    positions: dict[str, tuple[int, int]]  # href -> (x, y) of the node's centre
    width: int
    height: int

    @property
    def layers(self) -> tuple[int, int]:
        """The first and last layer of the circuit."""
        return min(u.layer for u in self.units), max(u.layer for u in self.units)

    @property
    def root(self) -> UnitNode:
        """The unit the circuit is named after: its first in layout order."""
        return self.units[0]

    @property
    def verified(self) -> int:
        """How many of the circuit's units stand VERIFIED."""
        return sum(u.verified for u in self.units)


def circuit_edges(model: ModelNode, rows: dict) -> list[CircuitEdge]:
    """Return the edges between two of ``model``'s units with pages.

    They come from every experiment's wiring. Each is found once per reading that names it.
    """
    units = {u.href: u for layer in model.layers for u in layer.units}
    edges: dict[tuple[str, str], CircuitEdge] = {}
    for unit in units.values():
        for e in unit.experiments:
            weights, activations, backfilled = record_wiring(e.record, rows)
            for wiring in (weights, activations):
                if wiring is not None:
                    seen = Reading(unit, e, wiring.method, 0.0, {}, backfilled)
                    _add_edges(edges, units, model.slug, seen, wiring)
    return sorted(
        edges.values(), key=lambda c: (c.source.layer, c.target.layer, c.source.href, c.target.href)
    )


def _add_edges(edges: dict, units: dict, slug: str, seen: Reading, wiring) -> None:
    """Add to ``edges`` each edge of one reading's ``wiring`` between two units with pages.

    ``seen`` is the reading with its unit, experiment, method and backfill. Each edge's
    reading takes its weight and channels.
    """
    sides = [
        (wiring.upstream_neurons, True),
        (wiring.upstream_heads or [], True),
        (wiring.downstream_neurons, False),
        (wiring.downstream_heads or [], False),
    ]
    unit = seen.unit
    for found, upstream in sides:
        for edge in found:
            other = units.get(HREF[edge.kind](slug, edge.layer, edge.index))
            if other is None or other is unit:
                continue
            source, target = (other, unit) if upstream else (unit, other)
            key = (source.href, target.href)
            if key not in edges:
                edges[key] = CircuitEdge(source, target)
            edges[key].readings.append(replace(seen, weight=edge.weight, channels=edge.channels))


@dataclass
class Hub:
    """A unit wired to so many others that its edges don't join circuits."""

    unit: UnitNode
    edges: list[CircuitEdge]  # every edge touching it, in layer order

    @property
    def neighbours(self) -> list[UnitNode]:
        """The units wired to the hub, in layer order, neurons before heads."""
        found = {}
        for e in self.edges:
            other = e.target if e.source is self.unit else e.source
            found[other.href] = other
        return sorted(found.values(), key=lambda u: (u.layer, not u.is_head, u.idx))


def hubs(model: ModelNode, edges: list[CircuitEdge]) -> list[Hub]:
    """Return the units wired to at least ``HUB_SHARE`` of ``model``'s units with pages.

    A hub also needs at least ``HUB_MIN`` edges. Most connected first.
    """
    n_units = sum(len(layer.units) for layer in model.layers)
    bar = max(HUB_MIN, HUB_SHARE * n_units)
    touching: dict[str, list[CircuitEdge]] = defaultdict(list)
    units = {}
    for e in edges:
        for u in (e.source, e.target):
            touching[u.href].append(e)
            units[u.href] = u
    found = [
        Hub(units[h], es)
        for h, es in touching.items()
        if len({o.href for e in es for o in (e.source, e.target)} - {h}) >= bar
    ]
    return sorted(found, key=lambda hub: (-len(hub.neighbours), hub.unit.href))


def circuits(model: ModelNode, rows: dict) -> list[Circuit]:
    """``model``'s circuits, largest first: groups of units joined by wiring edges."""
    return circuits_and_hubs(model, rows)[0]


def circuits_and_hubs(model: ModelNode, rows: dict) -> tuple[list[Circuit], list[Hub]]:
    """Return ``model``'s circuits, largest first, and its hubs.

    The circuits are joined by the edges that don't touch a hub.
    """
    edges = circuit_edges(model, rows)
    found_hubs = hubs(model, edges)
    hub_hrefs = {hub.unit.href for hub in found_hubs}
    edges = [e for e in edges if not {e.source.href, e.target.href} & hub_hrefs]
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
    return sorted(found, key=lambda c: (-len(c.units), c.layers, c.units[0].href)), found_hubs


def _layout(edges: list[CircuitEdge]) -> Circuit:
    """Columns by layer (heads before neurons, as attention runs first), rows within."""
    units = {u.href: u for e in edges for u in (e.source, e.target)}
    ordered = sorted(units.values(), key=lambda u: (u.layer, not u.is_head, u.idx, u.href))
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


def unit_pages(slug: str, found: list[Circuit], found_hubs: list[Hub]) -> dict[str, str]:
    """Return the page each unit appears on, by unit href.

    That is its circuit's page, else its own hub page, else the first hub it is wired to.
    """
    pages = {}
    for hub in found_hubs:
        for u in hub.neighbours:
            pages.setdefault(u.href, hub_href(slug, unit_key(hub.unit)))
    for hub in found_hubs:
        pages[hub.unit.href] = hub_href(slug, unit_key(hub.unit))
    for c in found:
        for u in c.units:
            pages[u.href] = circuit_href(slug, unit_key(c.root))
    return pages


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
