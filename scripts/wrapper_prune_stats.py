"""The tables of papers/wrapper-prune/README.md, from the rows of scripts/wrapper_prune.py (#5163).

For each pruned condition, it counts the poles that fail on the instruct model and pass again
(*recovered*). It also counts the poles that keep on the instruct model and fail once pruned
(*lost*). Top pruning and each random prune of the same K are compared on the same poles. The exact
McNemar test runs once for the recovered poles and once for the lost ones. It prints one
JSON line for each condition, and one for each layer's share of the top neurons.

    uv run python scripts/wrapper_prune_stats.py papers/wrapper-prune/results.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_change_stats import mcnemar  # noqa: E402


def pole_sets(rows: list[dict]) -> dict[tuple, dict[str, bool]]:
    """``{(condition, K, seed): {slug: keeps}}`` over the poles that reproduce on the base.

    Args:
        rows: The rows of ``scripts/wrapper_prune.py``.
    """
    out: dict[tuple, dict[str, bool]] = {}
    for r in rows:
        if r["table"] == "pole" and r["base_passes"]:
            out.setdefault((r["condition"], r["k"], r["seed"]), {})[r["slug"]] = r["keeps"]
    return out


def changes(rows: list[dict]) -> list[dict]:
    """For each pruned condition, its recovered and lost poles, against the instruct model.

    Args:
        rows: The rows of ``scripts/wrapper_prune.py``.

    Returns:
        One row for each condition. A random row also gives its McNemar tests against the top
        prune of the same K. ``recovered_p`` is on the failed poles, and ``lost_p`` on the kept.
    """
    sets = pole_sets(rows)
    none = sets["none", 0, None]
    failed = {s for s, keeps in none.items() if not keeps}
    kept = {s for s, keeps in none.items() if keeps}
    kl = {
        (r["condition"], r["k"], r["seed"]): r["held_kl"] for r in rows if r["table"] == "summary"
    }
    out = []
    for key in sorted(sets, key=lambda k: (k[1], k[0] != "top", k[2] or 0)):
        if key[0] == "none":
            continue
        got = sets[key]
        recovered = {s for s in failed if got[s]}
        lost = {s for s in kept if not got[s]}
        row = {"table": "change", "condition": key[0], "k": key[1], "seed": key[2],
               "failed_on_instruct": len(failed), "recovered": len(recovered),
               "kept_on_instruct": len(kept), "lost": len(lost), "held_kl": kl[key]}  # fmt: skip
        if key[0] == "random":
            top = sets["top", key[1], None]
            top_recovered = {s for s in failed if top[s]}
            top_lost = {s for s in kept if not top[s]}
            p_rec = mcnemar(len(top_recovered - recovered), len(recovered - top_recovered))
            p_lost = mcnemar(len(top_lost - lost), len(lost - top_lost))
            row["recovered_p"] = float(f"{p_rec:.3g}")  # three significant figures
            row["lost_p"] = float(f"{p_lost:.3g}")
        out.append(row)
    return out


def layers_of_top(rows: list[dict]) -> list[dict]:
    """For each K, how many of the top neurons sit in each layer, most first.

    Args:
        rows: The rows of ``scripts/wrapper_prune.py``.
    """
    return [
        {"table": "top_layers", "k": r["k"],
         "layers": Counter(layer for layer, _ in r["neurons"]).most_common()}
        for r in rows if r["table"] == "summary" and r["condition"] == "top"
    ]  # fmt: skip


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:
    """Print the recovered and lost poles of each condition, and the layers of the top neurons.

    Args:
        results: The rows of ``scripts/wrapper_prune.py``.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    for row in changes(rows) + layers_of_top(rows):
        click.echo(json.dumps(row))


if __name__ == "__main__":
    main()
