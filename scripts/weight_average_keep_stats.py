"""The tables of papers/weight-average-keep/README.md, from its results.jsonl (#5404).

It adds these rows to the results file's ``summary`` and ``pole`` rows:

* ``step``: for each α, the poles that keep under α·Δ and under the random change of its
  size (each seed), with the token KL, and how many of α = 1's flips also flip at α.
* ``paired``: exact McNemar tests on the paired poles. Each α is set against the random
  change of its size, and α = 0.5 and α = 0.75 against α = 1.
* ``path``: for each pole that flips at some α in (0, 1.5], its keep at every positive α,
  and whether its flips form one run that starts at some α and lasts to α = 1.5.

    uv run python scripts/weight_average_keep_stats.py papers/weight-average-keep/results.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_change_stats import mcnemar, overlap_p  # noqa: E402

TABLES = ("step", "paired", "path")


def flipped(rows: list[dict], condition: str, alpha: float, seed: int | None) -> set[str]:
    """The poles that reproduce on the base and fail under one condition.

    Args:
        rows: The results' rows.
        condition: ``interpolate`` or ``random``.
        alpha: The step.
        seed: The random change's seed, or None for ``interpolate``.

    Examples:
        >>> rows = [{"table": "pole", "condition": "interpolate", "alpha": 1.0, "seed": None,
        ...          "slug": s, "base_passes": True, "keeps": k} for s, k in (("a", 0), ("b", 1))]
        >>> flipped(rows, "interpolate", 1.0, None)
        {'a'}
    """
    return {r["slug"] for r in rows if r["table"] == "pole" and r["condition"] == condition
            and r["alpha"] == alpha and r["seed"] == seed and r["base_passes"]
            and not r["keeps"]}  # fmt: skip


def conditions(rows: list[dict]) -> list[tuple[str, float, int | None]]:
    """Each condition of the results, as ``(condition, alpha, seed)``, in the order run.

    Args:
        rows: The results' rows.
    """
    return [(r["condition"], r["alpha"], r["seed"]) for r in rows if r["table"] == "summary"]


def steps(rows: list[dict]) -> list[dict]:
    """The ``step`` rows: each condition's keeps, token KL and overlap with α = 1's flips.

    Args:
        rows: The results' rows.
    """
    summaries = {
        (r["condition"], r["alpha"], r["seed"]): r for r in rows if r["table"] == "summary"
    }
    tuned = flipped(rows, "interpolate", 1.0, None)
    out = []
    for key, s in summaries.items():
        flips = flipped(rows, *key)
        both = len(flips & tuned)
        total = s["reproduced_on_base"]
        out.append({"table": "step", "condition": key[0], "alpha": key[1], "seed": key[2],
                    "keeps": s["keeps"], "reproduced": total, "keep_share": s["keep_share"],
                    "keep_ci95": s["keep_ci95"], "flips": len(flips), "token_kl": s.get("token_kl"),
                    "change_norm": s["change_norm"], "shared_with_instruct": both,
                    "shared_p": round(overlap_p(both, total, len(flips), len(tuned)), 6),
                    "flipped": sorted(flips)})  # fmt: skip
    return out


def paired(rows: list[dict]) -> list[dict]:
    """The ``paired`` rows: exact McNemar tests between two conditions on the same poles.

    Args:
        rows: The results' rows.
    """
    keys = conditions(rows)
    pairs = [(("interpolate", a, None), k) for c, a, s in keys if c == "interpolate"
             for k in keys if k[0] == "random" and k[1] == a]  # fmt: skip
    pairs += [(("interpolate", a, None), ("interpolate", 1.0, None)) for a in (0.5, 0.75)]
    out = []
    for first, second in pairs:
        if first not in keys or second not in keys:
            continue
        a, b = flipped(rows, *first), flipped(rows, *second)
        out.append({"table": "paired", "first": list(first), "second": list(second),
                    "only_first": len(a - b), "only_second": len(b - a), "both": len(a & b),
                    "p": round(mcnemar(len(a - b), len(b - a)), 6)})  # fmt: skip
    return out


def one_run(keeps: list[bool]) -> bool:
    """True when the failures, in order of α, form one run that lasts to the last α.

    Args:
        keeps: Whether the pole keeps, at each positive α in increasing order.

    Examples:
        >>> one_run([True, True, False, False]), one_run([True, False, True, False])
        (True, False)
        >>> one_run([True, True, True, True])
        True
    """
    first = next((i for i, k in enumerate(keeps) if not k), len(keeps))
    return all(not k for k in keeps[first:])


def paths(rows: list[dict]) -> list[dict]:
    """The ``path`` rows: each pole that flips at some positive α, along the positive α.

    Args:
        rows: The results' rows.
    """
    alphas = sorted({a for c, a, s in conditions(rows) if c == "interpolate" and a > 0})
    keep = {(r["slug"], r["alpha"]): r["keeps"] for r in rows if r["table"] == "pole"
            and r["condition"] == "interpolate" and r["base_passes"]}  # fmt: skip
    slugs = sorted({s for s, a in keep if a in alphas and not keep[(s, a)]})
    out = []
    for slug in slugs:
        keeps = [keep[(slug, a)] for a in alphas]
        first = next(a for a, k in zip(alphas, keeps, strict=True) if not k)
        out.append({"table": "path", "slug": slug, "alphas": alphas, "keeps": keeps,
                    "first_fail": first, "one_run": one_run(keeps)})  # fmt: skip
    return out


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:  # pragma: no cover - reads and writes the results file
    """Add the step, paired and path rows to ``results``, and print them.

    Args:
        results: The results file of ``scripts/weight_average_keep.py``.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    rows = [r for r in rows if r["table"] not in TABLES]
    added = steps(rows) + paired(rows) + paths(rows)
    results.write_text("".join(json.dumps(r) + "\n" for r in rows + added))
    for row in added:
        click.echo(json.dumps({k: v for k, v in row.items() if k != "flipped"}))


if __name__ == "__main__":
    main()
