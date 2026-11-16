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

from openinterp.circuit_ids import MATCH, MIN_SIZE, TOUCH, clusters, jaccard  # noqa: E402,F401
from openinterp.circuit_ids import match as match_ids  # noqa: E402


def match(old: dict[str, frozenset], new: list[frozenset]) -> dict[int, tuple[str, float]]:
    """``{index of new circuit: (old id, overlap)}``, by the shared matching."""
    return {i: (cid, jaccard(new[i], old[cid])) for i, cid in match_ids(old, new).items()}


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
