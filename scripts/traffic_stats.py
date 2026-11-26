"""The tables of papers/traffic-map/README.md, from the rows of scripts/traffic_map.py (#5231).

For each predictor and each measure of disturbance, it prints Spearman's ρ over the tested
sites, with a bootstrap 95% interval over sites. A shuffle of the traffic across the tested
sites gives the null: the share of shuffles whose |ρ| reaches the observed |ρ|. A second
test ranks each variable within its kind and layer, and shuffles within them, so that a
predictor gets no credit for depth alone. The predictors are these:

* ``traffic``: Σ_p use_p · ‖∂R_p/∂θ_s‖.
* ``unweighted``: the same sum, with no use weight.
* ``count``: the number of poles whose gradient reaches the site.
* ``loss_fisher``: the diagonal Fisher of the loss over the site's weights.
* ``inverse_weight_norm``: the change's size relative to the site's own weights.

It also gives the known cases: the copy heads' ranks among the heads, and the traffic of
the dormant neurons. The correlations use the even draw (the decile neurons and every
head), and the top sites are summarised apart.

    uv run python scripts/traffic_stats.py papers/traffic-map/results.jsonl \
        papers/traffic-map/results.activity.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import click

PREDICTORS = ("traffic", "unweighted", "count", "loss_fisher", "inverse_weight_norm")
MEASURES = ("flips", "mean_drift")
COPY_HEADS = ((3, 3), (4, 0), (4, 7))  # the copy failure's heads (#4781)
DORMANT = 0.01  # a dormant pole is active on under 1% of tokens (scripts/dormant_rank.py)
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


def p_value(rho: float | None, at_least: int) -> float | None:
    """The shuffle p-value, with the observed order counted as one shuffle. None with no ρ.

    Args:
        rho: The observed ρ.
        at_least: The shuffles whose |ρ| reached the observed |ρ|.
    """
    return None if rho is None else round((at_least + 1) / (SHUFFLES + 1), 4)


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
                    "inverse_weight_norm": 1 / s["weight_norm"], "draw": t.get("draw"),
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
            "shuffle_p": p_value(rho, at_least)}  # fmt: skip


def difference(sites: list[dict], other: str, measure: str) -> dict:
    """ρ of traffic less ρ of another predictor, with a paired bootstrap 95% interval.

    Each replicate draws the sites once, and computes both ρ on the same draw.

    Args:
        sites: From :func:`tested_sites`.
        other: The predictor to compare with traffic.
        measure: One of :data:`MEASURES`.
    """
    y = [s[measure] for s in sites]
    a, b = [s["traffic"] for s in sites], [s[other] for s in sites]
    rng, diffs = random.Random(SEED), []
    for _ in range(BOOTSTRAPS):
        pick = [rng.randrange(len(y)) for _ in y]
        ra = spearman([a[i] for i in pick], [y[i] for i in pick])
        rb = spearman([b[i] for i in pick], [y[i] for i in pick])
        if ra is not None and rb is not None:
            diffs.append(ra - rb)
    diffs.sort()
    return {"table": "difference", "predictor": other, "measure": measure,
            "difference": round(spearman(a, y) - spearman(b, y), 4),
            "ci95": [round(diffs[int(0.025 * len(diffs))], 4),
                     round(diffs[int(0.975 * len(diffs)) - 1], 4)]}  # fmt: skip


def within_layer(sites: list[dict], predictor: str, measure: str) -> dict:
    """ρ after ranking each variable within its kind and layer, so that depth gets no credit.

    The null shuffles the predictor within each kind and layer.

    Args:
        sites: From :func:`tested_sites`.
        predictor: One of :data:`PREDICTORS`.
        measure: One of :data:`MEASURES`.
    """
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(sites):
        groups.setdefault((s["kind"], s["layer"]), []).append(i)

    def local(values: list[float]) -> list[float]:
        """Each value's rank within its group, as a share of the group, from 0 to 1.

        Args:
            values: One value for each site, in the order of ``sites``.
        """
        out = [0.5] * len(values)
        for members in groups.values():
            r = ranks([values[i] for i in members])
            for i, rank in zip(members, r, strict=True):
                out[i] = (rank - 1) / (len(members) - 1) if len(members) > 1 else 0.5
        return out

    x = [s[predictor] for s in sites]
    y = local([s[measure] for s in sites])
    rho = spearman(local(x), y)
    rng, at_least = random.Random(SEED), 0
    for _ in range(SHUFFLES):
        shuffled = x[:]
        for members in groups.values():
            values = [x[i] for i in members]
            rng.shuffle(values)
            for i, v in zip(members, values, strict=True):
                shuffled[i] = v
        r = spearman(local(shuffled), y)
        at_least += rho is not None and r is not None and abs(r) >= abs(rho)
    return {"table": "within_layer", "predictor": predictor, "measure": measure,
            "sites": len(sites), "rho": rho and round(rho, 4),
            "shuffle_p": p_value(rho, at_least)}  # fmt: skip


def known_cases(rows: list[dict], activity: list[dict]) -> dict:
    """The copy heads' ranks by traffic, and the dormant neurons' traffic against the rest.

    A neuron is dormant when both of its poles are active on under :data:`DORMANT` of the
    sample's tokens. The last layer's neurons are left out, because no pole lies above them.

    Args:
        rows: The rows of ``scripts/traffic_map.py``.
        activity: The rows of ``scripts/traffic_dormant.py``.
    """
    sites = [r for r in rows if r["table"] == "site"]
    heads = sorted((r for r in sites if r["kind"] == "head"), key=lambda r: -r["traffic"])
    rank = {(r["layer"], r["index"]): i + 1 for i, r in enumerate(heads)}
    within = {}
    for a, b in COPY_HEADS:
        layer = [r for r in heads if r["layer"] == a]
        within[f"L{a}/H{b}"] = [(r["layer"], r["index"]) for r in layer].index((a, b)) + 1
    shares = {(r["layer"], r["index"]): r for r in activity}
    last = max(r["layer"] for r in sites if r["kind"] == "mlp")
    neurons = [r for r in sites if r["kind"] == "mlp" and r["layer"] < last]
    order = sorted(neurons, key=lambda r: -r["traffic"])
    place = {(r["layer"], r["index"]): i + 1 for i, r in enumerate(order)}
    dormant = [r for r in neurons
               if max(shares[r["layer"], r["index"]][p] for p in ("positive", "negative"))
               < DORMANT]  # fmt: skip
    rest = [r["traffic"] for r in neurons if r not in dormant]
    return {"table": "known", "heads": len(heads),
            "copy_head_ranks": {f"L{a}/H{b}": rank[a, b] for a, b in COPY_HEADS},
            "copy_head_ranks_in_layer": within, "neurons": len(neurons),
            "dormant": [{"layer": r["layer"], "index": r["index"], "traffic": r["traffic"],
                         "rank": place[r["layer"], r["index"]]} for r in dormant],
            "median_traffic_rest": round(statistics.median(rest), 6)}  # fmt: skip


def draw_summary(sites: list[dict]) -> list[dict]:
    """Each draw's sites, flips and drift: the decile draw, the heads and the top sites.

    Args:
        sites: From :func:`tested_sites`.
    """
    out = []
    for draw in ("decile", "head", "top"):
        part = [s for s in sites if s["draw"] == draw]
        if part:
            out.append({"table": "draw", "draw": draw, "sites": len(part),
                        "flips": sum(s["flips"] for s in part),
                        "sites_with_flips": sum(s["flips"] > 0 for s in part),
                        "median_traffic": round(statistics.median(s["traffic"] for s in part), 6),
                        "median_drift": round(statistics.median(s["mean_drift"] for s in part), 6),
                        "max_flips": max(s["flips"] for s in part)})  # fmt: skip
    return out


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("activity", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path, activity: Path) -> None:
    """Print each draw, each predictor's correlation with each measure, then the known cases.

    The correlations use the even draw only: the decile neurons and every head.

    Args:
        results: The rows of ``scripts/traffic_map.py``.
        activity: The rows of ``scripts/traffic_dormant.py``.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    sites = tested_sites(rows)
    for row in draw_summary(sites):
        click.echo(json.dumps(row))
    even = [s for s in sites if s["draw"] in ("decile", "head")]
    for kind in ("all", "mlp", "head"):
        part = [s for s in even if kind in ("all", s["kind"])]
        for measure in MEASURES:
            for predictor in PREDICTORS:
                row = correlation(part, predictor, measure)
                click.echo(json.dumps({**row, "kind": kind}))
                if kind == "all":
                    click.echo(json.dumps(within_layer(part, predictor, measure)))
                if kind == "all" and predictor != "traffic":
                    click.echo(json.dumps(difference(part, predictor, measure)))
    shares = [json.loads(line) for line in activity.read_text().splitlines() if line]
    click.echo(json.dumps(known_cases(rows, shares)))


if __name__ == "__main__":
    sys.exit(main())
