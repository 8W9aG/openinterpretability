"""The tables of papers/lora-units/README.md, from the registry's records (#5166).

The study has two arms of 20 units each, audited by the same agents and the same review:

* ``adapter``: the ``down_proj`` directions of the rank-1 LoRA, from scripts/lora_units.py.
  Their records carry the method ``rank-one-lora``.
* ``native``: random native poles of the same model, drawn with seeds 5166 to 5185.

It writes these rows to ``results.jsonl``:

* ``unit``: one per audited unit, with its verdict, its metrics and the gates it failed.
* ``arm``: for each arm, the verdict counts, the VERIFIED share with a Wilson interval and
  the mean metrics.
* ``test``: Fisher's exact test of the VERIFIED share, adapters against natives. Also a
  seeded permutation test of the difference in mean specificity and in mean active share.
* ``gate``: for each arm and gate, how many units failed it.
* ``loading``: over the layers, how the adapters' ``|lora_A|`` sits on the VERIFIED neurons,
  from the ``loading`` rows of ``results.units.jsonl``.

    uv run python scripts/lora_units_stats.py --out papers/lora-units/results.jsonl
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.builder.learned_baseline import fisher_exact, wilson  # noqa: E402

MODEL_DIR = "Qwen--Qwen2.5-0.5B"
SEEDS = range(5166, 5186)  # the seeds of the native arm's random draws
ARMS = ("adapter", "native")
METRICS = ("specificity", "sensitivity", "balanced_accuracy", "active_share")
STATUSES = ("VERIFIED", "NEEDS_REVISION", "REJECTED")
PERMUTATIONS = 20_000
SEED = 5166


def arm_of(record: dict) -> str | None:
    """The study arm of one registry record, or None when it is not in the study.

    Args:
        record: One record, with ``submission`` and ``verdict``.

    Examples:
        >>> arm_of({"submission": {"superposition_profile": {"method": "rank-one-lora"}}})
        'adapter'
        >>> arm_of({"submission": {"provenance": {"selection": "random", "seed": 5170}}})
        'native'
        >>> arm_of({"submission": {"provenance": {"selection": "random", "seed": 7}}}) is None
        True
    """
    submission = record["submission"]
    if (submission.get("superposition_profile") or {}).get("method") == "rank-one-lora":
        return "adapter"
    provenance = submission.get("provenance") or {}
    if provenance.get("selection") == "random" and provenance.get("seed") in SEEDS:
        return "native"
    return None


def unit_row(record: dict, arm: str, path: str) -> dict:
    """The ``unit`` row of one record.

    Args:
        record: The record.
        arm: Its arm, from :func:`arm_of`.
        path: The record's path, relative to the repo.
    """
    ident, verdict = record["submission"]["identifier"], record["verdict"]
    metrics = verdict["metrics"]
    return {
        "table": "unit",
        "arm": arm,
        "layer": ident["layer"],
        "unit": ident.get("superposition_id") or f"N{ident['neuron_idx']}-{ident['pole']}",
        "seed": (record["submission"].get("provenance") or {}).get("seed"),
        "status": verdict["status"],
        "confidence": verdict["confidence_score"],
        **{m: metrics.get(m) for m in METRICS},
        "failed": sorted(g for g, ok in metrics["gates"].items() if not ok),
        "record": path,
    }


def collect(wiki: Path) -> list[dict]:
    """The ``unit`` rows of every study record in ``wiki``, the newest record per unit.

    Args:
        wiki: The registry's directory.
    """
    newest: dict[tuple, tuple[str, dict]] = {}
    for path in sorted((wiki / "experiments" / MODEL_DIR).glob("L*/*/*.json")):
        record = json.loads(path.read_text())
        arm = arm_of(record)
        if arm is None:
            continue
        row = unit_row(record, arm, path.as_posix())
        newest[(arm, row["layer"], row["unit"])] = (path.name, row)  # names sort by time
    return [row for _, row in sorted(newest.values(), key=lambda v: (v[1]["arm"], v[1]["layer"]))]


def mean(values: list) -> float | None:
    """The mean of the values that are not None, rounded, or None when there are none.

    Args:
        values: The values.

    Examples:
        >>> mean([1, None, 2])
        1.5
        >>> mean([None]) is None
        True
    """
    kept = [v for v in values if v is not None]
    return round(statistics.fmean(kept), 4) if kept else None


def arm_row(units: list[dict], arm: str) -> dict:
    """The ``arm`` row: verdict counts, the VERIFIED share and its interval, mean metrics.

    Args:
        units: The ``unit`` rows of every arm.
        arm: The arm to summarise.
    """
    rows = [u for u in units if u["arm"] == arm]
    verified = sum(u["status"] == "VERIFIED" for u in rows)
    low, high = wilson(verified, len(rows))
    return {
        "table": "arm",
        "arm": arm,
        "units": len(rows),
        **{s.lower(): sum(u["status"] == s for u in rows) for s in STATUSES},
        "verified_share": round(verified / len(rows), 4),
        "verified_interval": [round(low, 4), round(high, 4)],
        **{f"mean_{m}": mean([u[m] for u in rows]) for m in METRICS},
    }


def permutation_p(a: list[float], b: list[float], n: int = PERMUTATIONS, seed: int = SEED) -> float:
    """The two-sided permutation p-value of the difference in means of ``a`` and ``b``.

    Args:
        a: The first sample.
        b: The second sample.
        n: The number of random relabellings.
        seed: The seed of the relabellings.

    Examples:
        >>> permutation_p([0.0] * 8, [1.0] * 8, n=2000) < 0.01
        True
        >>> permutation_p([1.0, 2.0], [1.0, 2.0], n=200)
        1.0
    """
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    pooled, rng, extreme = a + b, random.Random(seed), 0
    for _ in range(n):
        rng.shuffle(pooled)
        diff = abs(statistics.fmean(pooled[: len(a)]) - statistics.fmean(pooled[len(a) :]))
        extreme += diff >= observed - 1e-12  # ties count as extreme
    return round((extreme + 1) / (n + 1), 6)  # the floor is 1/(n+1), so keep 6 places


def test_rows(units: list[dict]) -> list[dict]:
    """The ``test`` rows: adapters against natives, on the VERIFIED share and two metrics.

    Args:
        units: The ``unit`` rows of both arms.
    """
    by = {arm: [u for u in units if u["arm"] == arm] for arm in ARMS}
    k = {arm: sum(u["status"] == "VERIFIED" for u in by[arm]) for arm in ARMS}
    n = {arm: len(by[arm]) for arm in ARMS}
    fisher = fisher_exact(k["adapter"], n["adapter"], k["native"], n["native"])
    rows = [{"table": "test", "test": "fisher_exact", "measure": "verified_share",
             "p": round(fisher, 6)}]  # fmt: skip
    for metric in ("specificity", "active_share"):
        a, b = ([u[metric] for u in by[arm] if u[metric] is not None] for arm in ARMS)
        rows.append({"table": "test", "test": "permutation", "measure": metric,
                     "difference": round(statistics.fmean(a) - statistics.fmean(b), 4),
                     "p": permutation_p(a, b)})  # fmt: skip
    return rows


def gate_rows(units: list[dict]) -> list[dict]:
    """The ``gate`` rows: for each arm and each gate that any unit failed, the failures.

    Args:
        units: The ``unit`` rows.

    Examples:
        >>> gate_rows([{"arm": "adapter", "failed": ["specificity"]},
        ...            {"arm": "native", "failed": []}])
        [{'table': 'gate', 'gate': 'specificity', 'adapter': 1, 'native': 0}]
    """
    gates = sorted({g for u in units for g in u["failed"]})
    return [{"table": "gate", "gate": g,
             **{arm: sum(g in u["failed"] for u in units if u["arm"] == arm) for arm in ARMS}}
            for g in gates]  # fmt: skip


def sign_p(above: int, n: int) -> float:
    """The exact two-sided sign test p-value of ``above`` of ``n`` above the null's median.

    Args:
        above: How many values are above the median.
        n: How many values there are.

    Examples:
        >>> sign_p(5, 10)
        1.0
        >>> round(sign_p(10, 10), 4)
        0.002
    """
    tail = sum(math.comb(n, k) for k in range(min(above, n - above) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def loading_row(loadings: list[dict]) -> dict:
    """The ``loading`` row: where the VERIFIED neurons sit in each layer's ``|lora_A|``.

    Args:
        loadings: The rows of results.units.jsonl.

    Examples:
        >>> rows = [{"table": "loading", "verified": 2, "median_percentile": m,
        ...          "share_of_norm": 0.1, "share_expected": 0.05} for m in (0.4, 0.7, 0.9)]
        >>> got = loading_row(rows)
        >>> got["median_of_medians"], got["layers_above"], got["norm_ratio"]
        (0.7, 2, 2.0)
    """
    rows = [r for r in loadings if r["table"] == "loading" and r["median_percentile"] is not None]
    medians = [r["median_percentile"] for r in rows]
    above = sum(m > 0.5 for m in medians)
    return {
        "table": "loading",
        "layers": len(rows),
        "verified": sum(r["verified"] for r in rows),
        "median_of_medians": round(statistics.median(medians), 4),
        "layers_above": above,
        "sign_p": sign_p(above, len(rows)),
        "norm_ratio": round(
            sum(r["share_of_norm"] for r in rows) / sum(r["share_expected"] for r in rows), 4
        ),  # fmt: skip
    }


@click.command()
@click.option("--wiki", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", default="papers/lora-units/results.jsonl", type=click.Path(path_type=Path))
def main(wiki: Path, out: Path) -> None:  # pragma: no cover - reads the registry
    """Write the study's rows, and print the arm and test rows.

    Args:
        wiki: The registry's directory.
        out: The JSONL file to write.
    """
    units = collect(wiki)
    loadings = [json.loads(line) for line in (out.parent / "results.units.jsonl").open()]
    rows = units + [arm_row(units, arm) for arm in ARMS] + test_rows(units) + gate_rows(units)
    rows.append(loading_row(loadings))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for row in rows:
        if row["table"] != "unit":
            click.echo(json.dumps(row))


if __name__ == "__main__":
    main()
