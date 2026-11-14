"""Registry-wide statistics over random neuron poles (``TODO.md``).

The rates use only units chosen at random (``selection = "random"``). So they estimate
rates for the model's poles, not for the poles that an agent chose. The random draws that
the scan stopped on (``wiki/draws/``) are outcomes too. The tables count them next to the
audited units (``docs/analyses/registry-stats.md``).

    uv run open-interpretability build --output-dir build/site
    uv run python scripts/registry_stats.py --site-dir build/site \
        --out docs/analyses/registry-stats.jsonl
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import click
import duckdb

from openinterp.builder.learned_baseline import wilson
from openinterp.core.agent_names import canonical_agent
from openinterp.core.schemas import BUILTIN_CORPUS

STATUSES = ("VERIFIED", "NEEDS_REVISION", "REJECTED")
DEPTH_BANDS = 4  # bands of relative depth, to compare models with different layer counts
PERMUTATIONS = 10_000

# A unit's first reviewed experiment, with the columns the tables need. A sister study
# tests a borrowed explanation, so it is not an audit of the unit.
UNITS = """
SELECT slug, model_name, layer, pole, model_num_layers, selection, status, unit_status,
       submitting_agent_model, reviewer_agent_model, self_reviewed,
       epoch(reviewed_at) AS reviewed_at,  -- seconds: a TIMESTAMPTZ in Python needs pytz
       json_extract_string(record_json, '$.submission.provenance.corpus') AS corpus
FROM r
WHERE kind = 'mlp_neuron_pole' AND sister_of IS NULL
QUALIFY row_number() OVER (PARTITION BY slug ORDER BY reviewed_at, experiment_id) = 1
"""

# Each unit's experiments after the first: how many, and whether any one is VERIFIED.
FOLLOW_UPS = """
SELECT slug, count(*) AS experiments, bool_or(status = 'VERIFIED') AS ever_verified
FROM r WHERE kind = 'mlp_neuron_pole' AND sister_of IS NULL GROUP BY slug
"""

# Readings of one hypothesis: experiments on one unit with the same explanation, each
# with a probe set of its own. A re-measurement reuses probes, so it is not a reading.
READINGS = """
SELECT slug, explanation, status, reviewer_agent_model, submitting_agent_model,
       self_reviewed
FROM r
WHERE kind = 'mlp_neuron_pole' AND sister_of IS NULL AND reuses_probes_of IS NULL
ORDER BY slug, explanation, reviewed_at, experiment_id
"""

DRAWS = """
SELECT kind, model_name, layer, pole, outcome, active_share, epoch(drawn_at) AS drawn_at FROM d
"""


def load(site_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Return the random units, the readings of each hypothesis and the draws."""
    con = duckdb.connect()
    data = site_dir / "data"
    # The paths are local files the build wrote, not user input.
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{data / 'registry.parquet'}')")  # noqa: S608
    con.execute(f"CREATE VIEW d AS SELECT * FROM read_parquet('{data / 'draws.parquet'}')")  # noqa: S608
    follow = {row["slug"]: row for row in _rows(con, FOLLOW_UPS)}
    units = []
    for row in _rows(con, UNITS):
        if row["selection"] != "random":  # a follow-up's selection is "manual"; read the first
            continue
        row.update(follow[row["slug"]])
        row["proposer"] = canonical_agent(row["submitting_agent_model"])
        row["band"] = min(DEPTH_BANDS - 1, row["layer"] * DEPTH_BANDS // row["model_num_layers"])
        row["builtin"] = row["corpus"] == BUILTIN_CORPUS
        units.append(row)
    random_slugs = {u["slug"] for u in units}
    readings = [r for r in _rows(con, READINGS) if r["slug"] in random_slugs]
    draws = _rows(con, DRAWS)
    con.close()
    return units, readings, draws


def _rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def rates(units: list[dict], **group) -> dict:
    """Count verdicts for ``units``: first reading, standing and any reading VERIFIED."""
    n = len(units)
    first = Counter(u["status"] for u in units)
    standing = Counter(u["unit_status"] for u in units)
    ever = sum(u["ever_verified"] for u in units)
    row = {**group, "units": n, **{f"first_{s.lower()}": first[s] for s in STATUSES}}
    row |= {"standing_verified": standing["VERIFIED"], "contested": standing["CONTESTED"]}
    row |= {"ever_verified": ever, "experiments": sum(u["experiments"] for u in units)}
    if n:
        row["first_verified_ci"] = [round(x, 3) for x in wilson(first["VERIFIED"], n)]
        row["standing_verified_ci"] = [round(x, 3) for x in wilson(standing["VERIFIED"], n)]
    return row


def homogeneity_p(units: list[dict], key: str, seed: int = 0) -> float:
    """Permutation p that first-reading VERIFIED shares are equal across ``key``'s groups.

    The statistic is Pearson's chi-square on the groups × (VERIFIED, not) table. The
    permutations shuffle the verdicts over the units.
    """
    groups = [u[key] for u in units]
    hits = [u["status"] == "VERIFIED" for u in units]
    observed = _chi_square(groups, hits)
    rng = random.Random(seed)
    shuffled = hits[:]
    beats = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(shuffled)
        beats += _chi_square(groups, shuffled) >= observed - 1e-12
    return (beats + 1) / (PERMUTATIONS + 1)


def _chi_square(groups: list, hits: list[bool]) -> float:
    n, k = len(hits), sum(hits)
    if k in (0, n):
        return 0.0
    size, inside = Counter(groups), Counter(g for g, h in zip(groups, hits, strict=True) if h)
    stat = 0.0
    for g, m in size.items():
        for observed, share in ((inside[g], k / n), (m - inside[g], 1 - k / n)):
            expected = m * share
            stat += (observed - expected) ** 2 / expected
    return stat


def scan_stops(units: list[dict], draws: list[dict]) -> list[dict]:
    """Per model and pole, the share of random draws the scan stopped on.

    Only draws since the first recorded neuron-pole draw count. Before it a stopped scan
    left no file, so its share can't be known.
    """
    poles = [d for d in draws if d["kind"] == "mlp_neuron_pole"]
    if not poles:
        return []
    since = min(d["drawn_at"] for d in poles)
    audited = Counter((u["model_name"], u["pole"]) for u in units if u["reviewed_at"] >= since)
    stopped = defaultdict(Counter)
    for d in poles:
        stopped[d["model_name"], d["pole"]][d["outcome"]] += 1
    rows = []
    for model, pole in sorted(set(audited) | set(stopped)):
        dead, on = stopped[model, pole]["dead"], stopped[model, pole]["default_on"]
        drawn = audited[model, pole] + dead + on
        rows.append(
            {
                "table": "scan_stops",
                "model": model,
                "pole": pole,
                "since": datetime.fromtimestamp(since, UTC).isoformat(timespec="seconds"),
                "drawn": drawn,
                "audited": audited[model, pole],
                "dead": dead,
                "default_on": on,
                "stopped_share": round((dead + on) / drawn, 3) if drawn else None,
            }
        )
    return rows


def replications(readings: list[dict]) -> list[dict]:
    """Compare each later reading of a hypothesis with its first reading.

    Pairs are split by whether the later reviewer is the first reviewer's model.
    """
    by_hypothesis = defaultdict(list)
    for r in readings:
        by_hypothesis[r["slug"], r["explanation"]].append(r)
    pairs = defaultdict(Counter)
    for group in by_hypothesis.values():
        first, *later = group
        for r in later:
            same = canonical_agent(r["reviewer_agent_model"]) == canonical_agent(
                first["reviewer_agent_model"]
            )
            arm = "same_reviewer" if same else "other_reviewer"
            for name in (arm, "all"):
                pairs[name][first["status"], r["status"]] += 1
    rows = []
    for arm in ("all", "same_reviewer", "other_reviewer"):
        table = pairs[arm]
        n = sum(table.values())
        agree = sum(v for (a, b), v in table.items() if a == b)
        both_v = table["VERIFIED", "VERIFIED"]
        first_v = sum(v for (a, _), v in table.items() if a == "VERIFIED")
        later_v = sum(v for (a, b), v in table.items() if a != "VERIFIED" and b == "VERIFIED")
        rows.append(
            {
                "table": "replications",
                "arm": arm,
                "pairs": n,
                "status_agree": agree,
                "verified_agree": sum(
                    v for (a, b), v in table.items() if (a == "VERIFIED") == (b == "VERIFIED")
                ),
                "first_verified": first_v,
                "verified_then_verified": both_v,
                "first_not_verified": n - first_v,
                "not_verified_then_verified": later_v,
                "transitions": {f"{a}->{b}": v for (a, b), v in sorted(table.items())},
            }
        )
    return rows


def report(units: list[dict], readings: list[dict], draws: list[dict]) -> list[dict]:
    """Every table's rows, each with a ``table`` key."""
    rows = []
    for model in sorted({u["model_name"] for u in units}):
        mine = [u for u in units if u["model_name"] == model]
        rows.append(rates(mine, table="model", model=model))
        for pole in ("positive", "negative"):
            pole_units = [u for u in mine if u["pole"] == pole]
            rows.append(rates(pole_units, table="model_pole", model=model, pole=pole))
        for layer in sorted({u["layer"] for u in mine}):
            here = [u for u in mine if u["layer"] == layer]
            rows.append(rates(here, table="layer", model=model, layer=layer))
        if len({u["layer"] for u in mine}) > 1:
            p = homogeneity_p(mine, "layer")
            rows.append({"table": "layer_test", "model": model, "p": round(p, 4)})
        for builtin in (True, False):
            here = [u for u in mine if u["builtin"] == builtin]
            rows.append(rates(here, table="corpus", model=model, builtin_sample=builtin))
        for proposer in sorted({u["proposer"] for u in mine}):
            here = [u for u in mine if u["proposer"] == proposer]
            rows.append(rates(here, table="proposer", model=model, proposer=proposer))
            for builtin in (True, False):
                cell = [u for u in here if u["builtin"] == builtin]
                group = {"model": model, "proposer": proposer, "builtin_sample": builtin}
                rows.append(rates(cell, table="proposer_corpus", **group))
    for band in range(DEPTH_BANDS):
        here = [u for u in units if u["band"] == band]
        rows.append(rates(here, table="depth_band", band=band))
    rows.append({"table": "depth_band_test", "p": round(homogeneity_p(units, "band"), 4)})
    rows.append(rates(units, table="all"))
    rows += scan_stops(units, draws)
    rows += replications(readings)
    dead = Counter((d["kind"], d["outcome"]) for d in draws)
    rows.append({"table": "draws", **{f"{k}/{o}": v for (k, o), v in sorted(dead.items())}})
    return rows


@click.command()
@click.option(
    "--site-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("build/site")
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(site_dir: Path, out: Path) -> None:
    """Write one JSON row per table cell group to ``out``."""
    units, readings, draws = load(site_dir)
    rows = report(units, readings, draws)
    out.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))
    click.echo(f"{len(units)} random units, {len(draws)} draws → {out}", err=True)


if __name__ == "__main__":
    main()
