"""The paired tests of papers/random-change-keep/README.md, from its results.jsonl (#5154).

For each model, it compares the instruct change with each matched random change on the
same poles. It prints one JSON line for each comparison, with these values:

* The poles that flip under each change. Also the poles that flip under one change and not
  the other, with the exact McNemar p-value.
* The poles that flip under both changes, and the overlap that independent flips would
  give. Also the one-sided hypergeometric p-value of the observed overlap.
* The median activation correlation and scale ratio of each change.
* For each change, its flips in each quarter of the poles. The poles are in order of how
  far their base readings sit above the nearer gate.

    uv run python scripts/random_change_stats.py papers/random-change-keep/results.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from math import comb
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures_random_change_keep import margin  # noqa: E402

# Each instruct condition, and the random condition matched to it.
PAIRS = (("instruct", "random_all"), ("instruct_mlp", "random_mlp"))


def mcnemar(only_a: int, only_b: int) -> float:
    """The two-sided exact McNemar p-value of the discordant counts ``only_a`` and ``only_b``.

    Args:
        only_a: The poles that flip under the first change and not the second.
        only_b: The poles that flip under the second change and not the first.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def overlap_p(both: int, total: int, a: int, b: int) -> float:
    """The chance of an overlap of ``both`` or more, if ``a`` and ``b`` flips were independent.

    Args:
        both: The poles that flip under both changes.
        total: The poles that reproduce on the base.
        a: The poles that flip under the first change.
        b: The poles that flip under the second change.
    """
    top = min(a, b)
    hits = sum(comb(a, i) * comb(total - a, b - i) for i in range(both, top + 1))
    return hits / comb(total, b)


def flips_by_condition(rows: list[dict], model: str) -> tuple[list[str], dict]:
    """The poles that reproduce, and ``{(condition, seed): poles that flip}``.

    Args:
        rows: The paper's results.
        model: The base model's name.
    """
    poles = [r for r in rows if r["table"] == "pole" and r["model"] == model]
    reproduced = sorted({r["slug"] for r in poles if r["base_passes"]})
    out: dict[tuple, set[str]] = {}
    for r in poles:
        got = out.setdefault((r["condition"], r["seed"]), set())
        if r["base_passes"] and not r["keeps"]:
            got.add(r["slug"])
    return reproduced, out


def comparisons(rows: list[dict], model: str, pairs=PAIRS) -> list[dict]:
    """One row for each pair of a condition and one seed of its random condition.

    Args:
        rows: The paper's results.
        model: The base model's name.
        pairs: Each condition, and the random condition matched to it.
    """
    reproduced, flips = flips_by_condition(rows, model)
    out = []
    for ours, theirs in pairs:
        if (ours, None) not in flips:
            continue
        a = flips[ours, None]
        for (condition, seed), b in sorted(flips.items(), key=lambda kv: str(kv[0])):
            if condition != theirs:
                continue
            both = len(a & b)
            out.append({
                "table": "comparison", "model": model, "instruct": ours, "random": condition,
                "seed": seed, "reproduced": len(reproduced), "instruct_flips": len(a),
                "random_flips": len(b), "only_instruct": len(a - b), "only_random": len(b - a),
                "mcnemar_p": round(mcnemar(len(a - b), len(b - a)), 4), "both": both,
                "expected_both": round(len(a) * len(b) / len(reproduced), 2),
                "overlap_p": float(f"{overlap_p(both, len(reproduced), len(a), len(b)):.3g}"),
            })  # fmt: skip
    return out


def activity(rows: list[dict], model: str) -> list[dict]:
    """One row for each condition and seed: its median correlation and scale ratio.

    Args:
        rows: The paper's results.
        model: The base model's name.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["table"] == "pole" and r["model"] == model and r["base_passes"]:
            groups.setdefault((r["condition"], r["seed"]), []).append(r)
    out = []
    for (condition, seed), got in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rs = [r["pearson"] for r in got if r["pearson"] is not None]
        ratios = [r["ratio"] for r in got if r["ratio"] is not None]
        out.append({"table": "activity", "model": model, "condition": condition, "seed": seed,
                    "median_pearson": round(statistics.median(rs), 4),
                    "median_ratio": round(statistics.median(ratios), 4)})  # fmt: skip
    return out


def by_margin(rows: list[dict], model: str) -> list[dict]:
    """For each condition and seed, its flips in each quarter of the poles, nearest a gate first.

    Args:
        rows: The paper's results.
        model: The base model's name.
    """
    base = {r["slug"]: r["readings"] for r in rows if r["table"] == "base" and r["model"] == model}
    reproduced, flips = flips_by_condition(rows, model)
    order = sorted(reproduced, key=lambda s: (margin(base[s]), s))
    cuts = [round(len(order) * i / 4) for i in range(5)]
    quarters = [set(order[cuts[i] : cuts[i + 1]]) for i in range(4)]
    return [
        {"table": "margin", "model": model, "condition": condition, "seed": seed,
         "quarter_sizes": [len(q) for q in quarters],
         "flips_by_quarter": [len(got & q) for q in quarters]}
        for (condition, seed), got in sorted(flips.items(), key=lambda kv: str(kv[0]))
    ]  # fmt: skip


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:
    """Print the paired tests of each model in ``RESULTS``, one JSON line each.

    Args:
        results: The paper's results.jsonl.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    for model in sorted({r["model"] for r in rows if r["table"] == "pole"}):
        for row in comparisons(rows, model) + activity(rows, model) + by_margin(rows, model):
            click.echo(json.dumps(row))


if __name__ == "__main__":
    main()
