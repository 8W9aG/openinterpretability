"""The tables of papers/rome-keep/README.md, from the rows of scripts/rome_keep.py (#5167).

It prints one JSON line for each of these:

* ``check``: the readings of poles at or below the edit layer that changed. Since a ROME
  edit leaves every input weight as it is, any change there would be a bug.
* ``flips``: for ROME and for the random edits, the flips above the edit layer, by fact.
  Also the number of facts in which ROME flips fewer poles than its random edit.
* ``alignment``: the AUROC with which a pole's alignment with Λ predicts a flip. The null
  shuffles the outcomes within each layer and each edit, so that depth and the edit's size
  get no credit.
* ``margin``: the flips above the edit layer, by quarter of the poles' base margin.
* ``u_loading``: for each fact, the median percentile of |u_j| of the edit layer's
  VERIFIED neurons among all its neurons.

    uv run python scripts/rome_keep_stats.py papers/rome-keep/results.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures_random_change_keep import margin  # noqa: E402

SHUFFLES = 10_000
SEED = 5167


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    """The chance that a random positive scores above a random negative, ties half.

    Args:
        positives: The scores of the flips.
        negatives: The scores of the poles that keep.
    """
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def above(rows: list[dict], kind: str) -> list[dict]:
    """The pole rows of ``kind`` above the edit layer, of poles that reproduce on the base.

    Args:
        rows: The rows of ``scripts/rome_keep.py``.
        kind: ``rome`` or ``random``.
    """
    return [r for r in rows if r["table"] == "pole" and r["condition"] == kind
            and r["layer"] > r["edit_layer"] and r["base_passes"]]  # fmt: skip


def alignment_test(rows: list[dict], kind: str, shuffles: int = SHUFFLES) -> dict:
    """The AUROC of alignment for a flip, and its p-value under within-group shuffles.

    Args:
        rows: The rows of ``scripts/rome_keep.py``.
        kind: ``rome`` or ``random``.
        shuffles: The number of shuffles.
    """
    got = above(rows, kind)
    observed = auroc([r["alignment"] for r in got if not r["keeps"]],
                     [r["alignment"] for r in got if r["keeps"]])  # fmt: skip
    groups: dict[tuple, list[dict]] = {}
    for r in got:
        groups.setdefault((r["fact"], r["layer"]), []).append(r)
    rng, at_least = random.Random(SEED), 0
    for _ in range(shuffles):
        pos, neg = [], []
        # Each group keeps its number of flips, so depth and the edit's size carry over.
        for members in groups.values():
            outcomes = [m["keeps"] for m in members]
            rng.shuffle(outcomes)
            for m, keeps in zip(members, outcomes, strict=True):
                (neg if keeps else pos).append(m["alignment"])
        null = auroc(pos, neg)
        at_least += null is not None and observed is not None and null >= observed
    return {"table": "alignment", "condition": kind, "auroc": observed and round(observed, 4),
            "p": round((at_least + 1) / (shuffles + 1), 4), "shuffles": shuffles}  # fmt: skip


def flips(rows: list[dict]) -> dict:
    """The flips above the edit layer of each kind, by fact, and the facts where ROME flips fewer.

    Args:
        rows: The rows of ``scripts/rome_keep.py``.
    """
    out = {"table": "flips"}
    for kind in ("rome", "random"):
        got = above(rows, kind)
        by_fact: dict[int, int] = {}
        for r in got:
            by_fact[r["fact"]] = by_fact.get(r["fact"], 0) + (not r["keeps"])
        counts = [by_fact[f] for f in sorted(by_fact)]
        out[kind] = {"total": sum(counts), "by_fact": counts,
                     "distinct_poles": len({r["slug"] for r in got if not r["keeps"]}),
                     "median_pearson": round(statistics.median(
                         r["pearson"] for r in got if r["pearson"] is not None), 4)}  # fmt: skip
    pairs = zip(out["rome"]["by_fact"], out["random"]["by_fact"], strict=True)
    out["facts_rome_fewer"] = sum(a < b for a, b in pairs)
    return out


def check(rows: list[dict]) -> dict:
    """The readings at or below the edit layer that changed, which must be none.

    Args:
        rows: The rows of ``scripts/rome_keep.py``.
    """
    low = [r for r in rows if r["table"] == "pole" and r["layer"] <= r["edit_layer"]]
    moved = [r for r in low if (r["base_passes"] and not r["keeps"])
             or r["ratio"] not in (None, 1.0) or (r["pearson"] or 1.0) < 1 - 1e-6]  # fmt: skip
    return {"table": "check", "readings": len(low), "changed": len(moved)}


def margins(rows: list[dict]) -> dict:
    """The flips above the edit layer, by quarter of the poles' base margin, nearest a gate first.

    Args:
        rows: The rows of ``scripts/rome_keep.py``.
    """
    base = {r["slug"]: r["readings"] for r in rows if r["table"] == "base"}
    poles = sorted({r["slug"] for r in above(rows, "rome")}, key=lambda s: (margin(base[s]), s))
    cuts = [round(len(poles) * i / 4) for i in range(5)]
    quarter = {s: q for q in range(4) for s in poles[cuts[q] : cuts[q + 1]]}
    out = {"table": "margin", "quarter_sizes": [cuts[q + 1] - cuts[q] for q in range(4)]}
    for kind in ("rome", "random"):
        counts = [0, 0, 0, 0]
        for r in above(rows, kind):
            counts[quarter[r["slug"]]] += not r["keeps"]
        out[kind] = counts
    return out


def u_loadings(rows: list[dict]) -> dict:
    """The median percentile of the VERIFIED neurons' |u_j| for each edit, by kind.

    Args:
        rows: The rows of ``scripts/rome_keep.py``.
    """
    out = {"table": "u_loading"}
    for kind in ("rome", "random"):
        got = [r["u_loading"]["median_percentile"] for r in rows
               if r["table"] == "summary" and r["condition"] == kind]  # fmt: skip
        out[kind] = {"by_fact": got, "mean": round(statistics.mean(got), 4),
                     "above_half": sum(p > 0.5 for p in got)}  # fmt: skip
    return out


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:
    """Print the check, the flips, the alignment tests, the margins and the loadings.

    Args:
        results: The rows of ``scripts/rome_keep.py``.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    for row in (check(rows), flips(rows), alignment_test(rows, "rome"),
                alignment_test(rows, "random"), margins(rows), u_loadings(rows)):  # fmt: skip
        click.echo(json.dumps(row))


if __name__ == "__main__":
    main()
