"""The tables of papers/rome-sequence/README.md, from scripts/rome_sequence.py's rows (#5362).

It prints these rows:

* ``curve``: for each sequence and checkpoint, the poles above the edit layer that keep,
  with a Wilson interval. The row also gives the weight distance, the loss and the retention.
* ``paired``: at each checkpoint, ROME against the random sequence over the same poles, as
  discordant counts with an exact McNemar test.
* ``rate``: between checkpoints, the poles lost per unit of weight distance. A gradual
  decline has a steady rate, and a collapse at a threshold has one interval far above the
  rest.
* ``hazard``: one rate h for each sequence, fitted so that keeps ≈ N·exp(−h·distance), with
  the R² of the fit on the log scale. A constant hazard means a decline with no threshold.
* ``layers``: the keep share of each layer above the edit at each checkpoint.
* ``returns``: the poles that flip and then pass again at a later checkpoint.

    uv run python scripts/rome_sequence_stats.py papers/rome-sequence/results.jsonl
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_delta_keep import wilson  # noqa: E402

KINDS = ("rome", "random")


def mcnemar(b: int, c: int) -> float:
    """The exact two-sided McNemar p-value for ``b`` and ``c`` discordant pairs.

    Args:
        b: Pairs that keep under the first condition only.
        c: Pairs that keep under the second condition only.

    Examples:
        >>> mcnemar(0, 0)
        1.0
        >>> round(mcnemar(0, 6), 4)
        0.0312
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def above(rows: list[dict], layer: int) -> dict[tuple[str, int], dict[str, bool]]:
    """``{(kind, edits): {slug: keeps}}`` over the poles above ``layer`` that reproduce.

    Args:
        rows: The rows of ``scripts/rome_sequence.py``.
        layer: The edit layer.
    """
    out: dict[tuple[str, int], dict[str, bool]] = {}
    for r in rows:
        if r["table"] == "pole" and r["layer"] > layer and r["base_passes"]:
            out.setdefault((r["condition"], r["edits"]), {})[r["slug"]] = r["keeps"]
    return out


def curve(checkpoints: list[dict]) -> list[dict]:
    """Each checkpoint's keep count above the edit layer, with its Wilson interval.

    Args:
        checkpoints: The ``checkpoint`` rows.
    """
    return [{"table": "curve", "condition": c["condition"], "edits": c["edits"],
             "keeps_above": c["keeps_above"], "poles_above": c["poles_above"],
             "ci95": wilson(c["keeps_above"], c["poles_above"]),
             "keeps_below": c["keeps_below"], "distance": c["distance"], "loss": c["loss"],
             "retention": c["retention"], "latest": c["latest"]}
            for c in checkpoints]  # fmt: skip


def paired(keeps: dict, counts: list[int]) -> list[dict]:
    """ROME against random at each checkpoint, over the same poles.

    Args:
        keeps: From :func:`above`.
        counts: The checkpoints' edit counts.
    """
    out = []
    for n in counts:
        rome, rand = keeps[("rome", n)], keeps[("random", n)]
        b = sum(rome[s] and not rand[s] for s in rome)
        c = sum(rand[s] and not rome[s] for s in rome)
        out.append({"table": "paired", "edits": n, "rome_only": b, "random_only": c,
                    "p": round(mcnemar(b, c), 6)})  # fmt: skip
    return out


def rate(checkpoints: list[dict]) -> list[dict]:
    """Poles lost per unit of weight distance, between each pair of adjacent checkpoints.

    Args:
        checkpoints: The ``checkpoint`` rows of one sequence, in order of edits.
    """
    out = []
    for a, b in zip(checkpoints, checkpoints[1:], strict=False):
        step = b["distance"] - a["distance"]
        lost = a["keeps_above"] - b["keeps_above"]
        out.append({"table": "rate", "condition": a["condition"], "from": a["edits"],
                    "to": b["edits"], "distance_step": round(step, 4), "lost": lost,
                    "per_distance": round(lost / step, 3) if step > 0 else None})  # fmt: skip
    return out


def hazard(checkpoints: list[dict]) -> dict:
    """One rate h with keeps ≈ N·exp(−h·distance), by least squares on the log scale.

    The fit goes through the base point, where nothing has flipped. Checkpoints where no
    pole keeps are left out, because their log is not finite.

    Args:
        checkpoints: The ``checkpoint`` rows of one sequence, in order of edits.

    Returns:
        The rate, its R² and the number of checkpoints fitted.

    Examples:
        >>> rows = [{"condition": "x", "distance": d, "keeps_above": round(80 * math.exp(-0.5 * d)),
        ...          "poles_above": 80} for d in (0, 1, 2, 4)]
        >>> round(hazard(rows)["h"], 2)
        0.5
    """
    n0 = checkpoints[0]["poles_above"]
    pts = [(c["distance"], math.log(c["keeps_above"] / n0)) for c in checkpoints[1:]
           if c["keeps_above"] > 0]  # fmt: skip
    h = -sum(d * y for d, y in pts) / sum(d * d for d, _ in pts)
    mean = sum(y for _, y in pts) / len(pts)
    resid = sum((y + h * d) ** 2 for d, y in pts)
    total = sum((y - mean) ** 2 for _, y in pts)
    return {"table": "hazard", "condition": checkpoints[0]["condition"], "h": round(h, 4),
            "r2": round(1 - resid / total, 4) if total else None, "points": len(pts)}  # fmt: skip


def hazard_interval(keeps: dict, checkpoints: list[dict], draws: int = 2000,
                    seed: int = 5362) -> list[float]:  # fmt: skip
    """A bootstrap 95% interval of :func:`hazard`'s rate, resampling the poles.

    Args:
        keeps: From :func:`above`.
        checkpoints: The ``checkpoint`` rows of one sequence, in order of edits.
        draws: The bootstrap draws.
        seed: The seed of the draws.
    """
    kind = checkpoints[0]["condition"]
    slugs = sorted(keeps[(kind, checkpoints[0]["edits"])])
    rng, rates = random.Random(seed), []
    for _ in range(draws):
        pick = [rng.choice(slugs) for _ in slugs]
        rows = [{**c, "keeps_above": sum(keeps[(kind, c["edits"])][s] for s in pick),
                 "poles_above": len(pick)} for c in checkpoints]  # fmt: skip
        rates.append(hazard(rows)["h"])
    rates.sort()
    return [rates[int(0.025 * draws)], rates[int(0.975 * draws) - 1]]


def layers(rows: list[dict], layer: int) -> list[dict]:
    """The keep share of each layer above the edit, at each checkpoint of each sequence.

    Args:
        rows: The rows of ``scripts/rome_sequence.py``.
        layer: The edit layer.
    """
    tally: dict[tuple, list[bool]] = {}
    for r in rows:
        if r["table"] == "pole" and r["layer"] > layer and r["base_passes"]:
            tally.setdefault((r["condition"], r["edits"], r["layer"]), []).append(r["keeps"])
    return [{"table": "layers", "condition": k, "edits": n, "layer": lay,
             "keeps": sum(v), "poles": len(v)}
            for (k, n, lay), v in sorted(tally.items())]  # fmt: skip


def returns(keeps: dict, counts: list[int]) -> list[dict]:
    """For each sequence, the poles that flip at one checkpoint and pass at a later one.

    Args:
        keeps: From :func:`above`.
        counts: The checkpoints' edit counts, in order.
    """
    out = []
    for kind in KINDS:
        slugs = keeps[(kind, counts[0])]
        back = [s for s in slugs if any(
            not keeps[(kind, a)][s] and keeps[(kind, b)][s]
            for i, a in enumerate(counts) for b in counts[i + 1:])]  # fmt: skip
        out.append({"table": "returns", "condition": kind, "poles": len(back),
                    "of": len(slugs)})  # fmt: skip
    return out


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:
    """Print every table's rows as JSON lines.

    Args:
        results: The rows of ``scripts/rome_sequence.py``.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    checkpoints = [r for r in rows if r["table"] == "checkpoint"]
    layer = checkpoints[0]["edit_layer"]
    counts = sorted({c["edits"] for c in checkpoints})
    keeps = above(rows, layer)
    out = curve(checkpoints) + paired(keeps, counts)
    for kind in KINDS:
        mine = [c for c in checkpoints if c["condition"] == kind]
        out += [*rate(mine), {**hazard(mine), "ci95": hazard_interval(keeps, mine)}]
    out += layers(rows, layer) + returns(keeps, counts)
    for row in out:
        click.echo(json.dumps(row))


if __name__ == "__main__":
    main()
