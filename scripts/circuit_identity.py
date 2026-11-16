"""Do circuit ids survive as the registry grows? (docs/circuits-and-hubs.md, issue #4283).

Replays the registry day by day. Each day it builds the tree from the records that existed
then, and clusters each model's wiring graph with Leiden, leaving out hub edges. Each
cluster of at least 3 units is a circuit. Then it matches the day's circuits to the day
before by Jaccard overlap, with a maximum-weight matching. A circuit keeps its id when its
match overlaps by at least MATCH. Each day's events and the last day's circuits go to
``--out``::

    uv run python scripts/circuit_identity.py --out docs/analyses/circuit-identity.jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import click
import networkx as nx

MIN_SIZE = 3  # units in a circuit
MATCH = 0.5  # Jaccard overlap at which a circuit keeps its id
TOUCH = 0.2  # overlap at which a split or merge is counted
SEED = 0


def jaccard(a: frozenset, b: frozenset) -> float:
    """|A ∩ B| / |A ∪ B|: 1 minus the set edit distance scaled by the union."""
    return len(a & b) / len(a | b) if a | b else 1.0


def clusters(model, rows: dict) -> list[frozenset]:
    """The model's circuits: Leiden communities of its non-hub wiring edges, of 3+ units."""
    from openinterp.builder.circuits import circuit_edges, hubs

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
    return sorted((frozenset(c) for c in found if len(c) >= MIN_SIZE), key=lambda c: sorted(c))


def match(old: dict[str, frozenset], new: list[frozenset]) -> dict[int, tuple[str, float]]:
    """Pair each new circuit with the old one it overlaps most, by maximum-weight matching.

    Returns ``{index of new circuit: (old id, overlap)}`` for pairs at :data:`MATCH` or more.
    """
    graph = nx.Graph()
    for i, members in enumerate(new):
        for cid, before in old.items():
            if (overlap := jaccard(members, before)) >= MATCH:
                graph.add_edge(("new", i), ("old", cid), weight=overlap)
    pairs = {}
    for a, b in nx.max_weight_matching(graph):
        n, o = (a, b) if a[0] == "new" else (b, a)
        pairs[n[1]] = (o[1], graph[a][b]["weight"])
    return pairs


def step(
    old: dict[str, frozenset], new: list[frozenset], next_id: int
) -> tuple[dict, Counter, int]:
    """Give ``new`` circuits their ids from ``old``, and count the day's events."""
    paired, events, named = match(old, new), Counter(), {}
    for i, members in enumerate(new):
        if i in paired:
            cid = paired[i][0]
            events["continue" if members == old[cid] else "grow_or_shrink"] += 1
        else:
            cid, next_id = f"C{next_id}", next_id + 1
            events["new"] += 1
        named[cid] = members
    for before in old.values():  # an old circuit touching several new ones split
        if sum(jaccard(before, m) >= TOUCH for m in new) >= 2:
            events["split"] += 1
    for members in new:  # a new circuit touching several old ones merged them
        if sum(jaccard(members, b) >= TOUCH for b in old.values()) >= 2:
            events["merge"] += 1
    events["gone"] = sum(cid not in named for cid in old)
    return named, events, next_id


@click.command()
@click.option("--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("wiki"))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(wiki_dir: Path, out: Path) -> None:
    """Replay the registry day by day and record how circuit ids change."""
    from openinterp.builder.tree import build_tree
    from openinterp.builder.wiki import load_archive
    from openinterp.core.errata import load_errata
    from openinterp.wiring_backfill import load_rows

    records, errata, rows = load_archive(wiki_dir), load_errata(wiki_dir), load_rows()
    times = sorted(r.verdict.timestamp for r in records)
    day, last = times[0].replace(hour=23, minute=59, second=59), times[-1]
    state: dict[str, dict[str, frozenset]] = {}
    next_id, out_rows = 1, []
    while day <= last + timedelta(days=1):
        shown = [r for r in records if r.verdict.timestamp <= day]
        for model in build_tree(shown, errata):
            new = clusters(model, rows)
            named, events, next_id = step(state.get(model.slug, {}), new, next_id)
            state[model.slug] = named
            out_rows.append({"table": "day", "day": day.date().isoformat(), "model": model.slug,
                             "records": len(shown), "circuits": len(new), **events})  # fmt: skip
        click.echo(f"{day.date()}: {len(shown)} records", err=True)
        day += timedelta(days=1)
    for model, named in state.items():
        for cid, members in sorted(named.items()):
            out_rows.append({"table": "circuit", "model": model, "id": cid,
                             "members": sorted(members)})  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    click.echo(f"wrote {len(out_rows)} rows → {out}", err=True)


if __name__ == "__main__":
    main()
