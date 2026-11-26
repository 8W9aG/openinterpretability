"""The tables of papers/traffic-map/README.md, from the rows of scripts/traffic_map.py (#5231).

For each predictor and each measure of disturbance, it prints Spearman's ρ over the tested
sites, with a bootstrap 95% interval over sites. A shuffle of the traffic across the tested
sites gives the null: the share of shuffles whose |ρ| reaches the observed |ρ|. The
predictors are these:

* ``traffic``: Σ_p use_p · ‖∂R_p/∂θ_s‖.
* ``unweighted``: the same sum, with no use weight.
* ``count``: the number of poles whose gradient reaches the site.
* ``loss_fisher``: the diagonal Fisher of the loss over the site's weights.
* ``inverse_weight_norm``: the change's size relative to the site's own weights.

It also gives the known cases: the copy heads' ranks among the heads, and the traffic of
the least active neurons.

    uv run python scripts/traffic_stats.py papers/traffic-map/results.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click

PREDICTORS = ("traffic", "unweighted", "count", "loss_fisher", "inverse_weight_norm")
MEASURES = ("flips", "mean_drift")
COPY_HEADS = ((3, 3), (4, 0), (4, 7))  # the copy failure's heads (#4781)
SHUFFLES = 10_000
BOOTSTRAPS = 2_000
SEED = 5231


def ranks(values: list[float]) -> list[float]:
    """The rank of each value, from 1, with ties given their mean rank.

    Args:
        values: The values.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman's ρ: Pearson's r of the ranks. None when either side is constant.

    Args:
        x: One variable.
        y: The other, in the same order.
    """
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / (sx * sy)


def tested_sites(rows: list[dict]) -> list[dict]:
    """Each tested site's predictors and measures, joined from its ``site`` and ``test`` rows.

    Args:
        rows: The rows of ``scripts/traffic_map.py``.
    """
    sites = {(r["kind"], r["layer"], r["index"]): r for r in rows if r["table"] == "site"}
    out = []
    for t in (r for r in rows if r["table"] == "test"):
        s = sites[t["kind"], t["site_layer"], t["index"]]
        out.append({**{k: s[k] for k in ("kind", "layer", "index", "traffic", "unweighted",
                                         "count", "loss_fisher")},
                    "inverse_weight_norm": 1 / s["weight_norm"],
                    "flips": t["flips"], "mean_drift": t["mean_drift"]})  # fmt: skip
    return out


def correlation(sites: list[dict], predictor: str, measure: str) -> dict:
    """ρ of one predictor with one measure, its bootstrap interval and its shuffle p-value.

    Args:
        sites: From :func:`tested_sites`.
        predictor: One of :data:`PREDICTORS`.
        measure: One of :data:`MEASURES`.
    """
    x, y = [s[predictor] for s in sites], [s[measure] for s in sites]
    rho = spearman(x, y)
    rng = random.Random(SEED)
    boots = []
    for _ in range(BOOTSTRAPS):
        pick = [rng.randrange(len(x)) for _ in x]
        r = spearman([x[i] for i in pick], [y[i] for i in pick])
        if r is not None:
            boots.append(r)
    boots.sort()
    at_least = 0
    for _ in range(SHUFFLES):
        shuffled = x[:]
        rng.shuffle(shuffled)
        r = spearman(shuffled, y)
        at_least += rho is not None and r is not None and abs(r) >= abs(rho)
    low, high = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots \
        else (None, None)  # fmt: skip
    return {"table": "correlation", "predictor": predictor, "measure": measure,
            "sites": len(sites), "rho": rho and round(rho, 4),
            "ci95": [low and round(low, 4), high and round(high, 4)],
            "shuffle_p": round((at_least + 1) / (SHUFFLES + 1), 4)}  # fmt: skip


def known_cases(rows: list[dict]) -> dict:
    """The copy heads' ranks by traffic among all heads, and the quiet neurons' traffic.

    Args:
        rows: The rows of ``scripts/traffic_map.py``.
    """
    heads = sorted((r for r in rows if r["table"] == "site" and r["kind"] == "head"),
                   key=lambda r: -r["traffic"])  # fmt: skip
    rank = {(r["layer"], r["index"]): i + 1 for i, r in enumerate(heads)}
    return {"table": "known", "heads": len(heads),
            "copy_head_ranks": {f"L{a}/H{b}": rank[a, b] for a, b in COPY_HEADS}}  # fmt: skip


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:
    """Print each predictor's correlation with each measure, then the known cases.

    Args:
        results: The rows of ``scripts/traffic_map.py``.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    sites = tested_sites(rows)
    for kind in ("all", "mlp", "head"):
        part = [s for s in sites if kind in ("all", s["kind"])]
        for measure in MEASURES:
            for predictor in PREDICTORS:
                row = correlation(part, predictor, measure)
                click.echo(json.dumps({**row, "kind": kind}))
    click.echo(json.dumps(known_cases(rows)))


if __name__ == "__main__":
    sys.exit(main())
