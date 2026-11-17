"""Circuit ids that survive as the registry grows (docs/circuits-and-hubs.md, issue #4284).

A circuit is a cluster of a model's wiring graph, of at least :data:`MIN_SIZE` units. The
clusters are Leiden communities of the direct-effect edges between studied units. Edges
are weighted by strength, and hub edges are left out. Each run matches the clusters to
the circuits stored in ``data/circuits/<model slug>.jsonl`` by member overlap (Jaccard),
with a maximum-weight matching. A cluster keeps the id of the circuit that it matches
at :data:`MATCH` or more, else it gets a new id. The file is append-only. Each line is
one event: ``new``, ``grow`` (members changed), ``split``, ``merge`` or ``gone``. A
circuit's members are those of its newest event. ``open-interpretability circuits
identify`` runs it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

CIRCUITS_DIR = Path("data/circuits")
MIN_SIZE = 3  # units in a circuit
MATCH = 0.5  # Jaccard overlap at which a circuit keeps its id
TOUCH = 0.2  # overlap at which a split or a merge is recorded
SEED = 0  # Leiden's seed, so one graph always gives one clustering


def jaccard(a: frozenset, b: frozenset) -> float:
    """|A ∩ B| / |A ∪ B|: 1 minus the set edit distance between A and B, scaled by the union."""
    return len(a & b) / len(a | b) if a | b else 1.0


def clusters(model, rows: dict) -> list[frozenset]:
    """The model's clusters: Leiden communities of its non-hub wiring edges, of 3+ units.

    Each unit is named by its page (``models/<slug>/L3/N7.html``).
    """
    from openinterpretability.builder.circuits import circuit_edges, hubs

    edges = circuit_edges(model, rows)
    hub_units = {h.unit.href for h in hubs(model, edges)}
    graph = nx.Graph()
    for e in edges:
        a, b = e.source.href, e.target.href
        if a in hub_units or b in hub_units:
            continue
        weight = sum(abs(r.weight or 0.0) for r in e.readings) / len(e.readings) or 1e-6
        old = graph.get_edge_data(a, b, {"weight": 0.0})["weight"]
        graph.add_edge(a, b, weight=old + weight)
    if graph.number_of_edges() == 0:
        return []
    found = nx.community.leiden_communities(graph, weight="weight", seed=SEED)
    return sorted((frozenset(c) for c in found if len(c) >= MIN_SIZE), key=sorted)


def match(old: dict[str, frozenset], new: list[frozenset]) -> dict[int, str]:
    """``{index of a new cluster: the id of the old circuit it continues}``.

    Pairs overlap by at least :data:`MATCH`, and a maximum-weight matching chooses them.
    """
    graph = nx.Graph()
    for i, members in enumerate(new):
        for cid, before in old.items():
            if (overlap := jaccard(members, before)) >= MATCH:
                graph.add_edge(("new", i), ("old", cid), weight=overlap)
    pairs = {}
    for a, b in nx.max_weight_matching(graph):
        n, o = (a, b) if a[0] == "new" else (b, a)
        pairs[n[1]] = o[1]
    return pairs


@dataclass
class Update:
    """One run's result: the circuits now, and the events to append."""

    circuits: dict[str, frozenset]
    events: list[dict] = field(default_factory=list)


def _next_id(taken) -> int:
    numbers = [int(cid[1:]) for cid in taken if cid[1:].isdigit()]
    return max(numbers, default=0) + 1


def update(old: dict[str, frozenset], new: list[frozenset], date: str, ever=()) -> Update:
    """Name ``new`` clusters from ``old`` circuits, and list the events of the change.

    ``ever`` holds every id the model has used, so a new id is never one that came before.
    """
    paired, out, events = match(old, new), {}, []
    counter = _next_id({*old, *ever})
    for i, members in enumerate(new):
        if i in paired:
            cid = paired[i]
            if members != old[cid]:
                grown = {"date": date, "event": "grow", "id": cid, "members": sorted(members)}
                events.append(grown)
        else:
            cid, counter = f"C{counter}", counter + 1
            events.append({"date": date, "event": "new", "id": cid, "members": sorted(members)})
        out[cid] = members
    for cid, before in old.items():
        parts = sorted(n for n, m in out.items() if jaccard(before, m) >= TOUCH and n != cid)
        if parts and (cid in out or len(parts) >= 2):
            events.append({"date": date, "event": "split", "id": cid, "into": parts})
    for cid, members in out.items():
        sources = sorted(o for o, b in old.items() if jaccard(members, b) >= TOUCH and o != cid)
        if sources and (cid in old or len(sources) >= 2):
            events.append({"date": date, "event": "merge", "id": cid, "from": sources})
    for cid in old:
        if cid not in out:
            events.append({"date": date, "event": "gone", "id": cid})
    return Update(out, events)


def path_for(model_slug: str, folder: Path = CIRCUITS_DIR) -> Path:
    """The model's lineage file."""
    return folder / f"{model_slug}.jsonl"


def load(path: Path) -> tuple[dict[str, frozenset], set[str]]:
    """The circuits as the file leaves them, and every id it has used."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    events = [json.loads(line) for line in lines if line.strip()]
    return load_events(events), {event["id"] for event in events}


def load_events(events: list[dict]) -> dict[str, frozenset]:
    """The circuits that ``events``, in file order, leave: each id's newest members."""
    circuits: dict[str, frozenset] = {}
    for event in events:
        if event["event"] == "gone":
            circuits.pop(event["id"], None)
        elif "members" in event:
            circuits[event["id"]] = frozenset(event["members"])
    return circuits


def append(path: Path, events: list[dict]) -> None:
    """Append ``events`` to the model's lineage file, one JSON object per line."""
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.writelines(json.dumps(e, sort_keys=True) + "\n" for e in events)
