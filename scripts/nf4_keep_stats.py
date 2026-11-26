"""The analyses of papers/nf4-keep/README.md, from the rows of scripts/nf4_keep.py (#5158).

``drift`` prints, for each model and condition, the share of probes whose peak drifts by
less than each mark, and the 99% and 99.9% points, as
``docs/analyses/half-precision-reproduce.md`` gives them.

``paired`` compares NF4 with each seed of its matched random change, on the same poles.

``normality`` runs Appendix F's test of the QLoRA paper (Dettmers et al., 2023). It is a
Shapiro-Wilk test of each VERIFIED pole's weights: its ``gate_proj`` and ``up_proj`` rows
and its ``down_proj`` column, on the base model, at the 5% level. It writes one row for
each pole. It then prints whether the poles with a non-normal weight lose their explanation
under NF4 more often, with Fisher's exact test. It needs SciPy::

    uv run python scripts/nf4_keep_stats.py drift papers/nf4-keep/results.jsonl
    uv run python scripts/nf4_keep_stats.py paired papers/nf4-keep/results.jsonl \\
        --instruct papers/random-change-keep/results.jsonl
    uv run --with scipy==1.16.3 python scripts/nf4_keep_stats.py normality \\
        papers/nf4-keep/results.jsonl --out normality.jsonl
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

ALPHA = 0.05  # Appendix F's level
WEIGHTS = ("gate_proj", "up_proj", "down_proj")
MARKS = (0.01, 0.03, 0.05, 0.07)  # drift marks, as shares of the pole's scale


def quantile(values: list[float], q: float) -> float:
    """The ``q`` quantile of ``values``, with linear interpolation between ranks.

    Args:
        values: The values, in any order.
        q: The quantile, from 0 to 1.

    Returns:
        The quantile.
    """
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def drift_rows(rows: list[dict]) -> list[dict]:
    """For each model, condition and seed, the drift of every probe of the poles that reproduce.

    Args:
        rows: The rows of ``scripts/nf4_keep.py``.

    Returns:
        One row for each model, condition and seed.
    """
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        if r["table"] == "pole" and r["base_passes"]:
            groups.setdefault((r["model"], r["condition"], r["seed"]), []).extend(r["drift"])
    return [
        {"table": "drift", "model": m, "condition": c, "seed": s, "probes": len(d),
         "under": {str(mark): round(sum(x < mark for x in d) / len(d), 4) for mark in MARKS},
         "q99": round(quantile(d, 0.99), 4), "q999": round(quantile(d, 0.999), 4),
         "largest": round(max(d), 4)}
        for (m, c, s), d in sorted(groups.items(), key=lambda kv: str(kv[0]))
    ]  # fmt: skip


def fisher(a: int, b: int, c: int, d: int) -> float:
    """The two-sided p-value of Fisher's exact test of the table ``[[a, b], [c, d]]``.

    Args:
        a: Row 1, column 1.
        b: Row 1, column 2.
        c: Row 2, column 1.
        d: Row 2, column 2.

    Returns:
        The sum of the chances of every table with the same margins that is no likelier.
    """
    row, col, n = a + b, a + c, a + b + c + d

    def chance(x: int) -> float:
        return comb(col, x) * comb(n - col, row - x) / comb(n, row)

    seen = chance(a)
    tables = range(max(0, row + col - n), min(row, col) + 1)
    return min(1.0, sum(p for x in tables if (p := chance(x)) <= seen * (1 + 1e-9)))


def weight_vectors(model, layer: int, neuron: int) -> dict:
    """The pole's three weight vectors: its gate and up rows, and its down column.

    Args:
        model: A Qwen2 or Llama model.
        layer: The layer.
        neuron: The neuron.

    Returns:
        ``{name: 1-D tensor}`` for each of :data:`WEIGHTS`.
    """
    mlp = model.model.layers[layer].mlp
    return {"gate_proj": mlp.gate_proj.weight[neuron], "up_proj": mlp.up_proj.weight[neuron],
            "down_proj": mlp.down_proj.weight[:, neuron]}  # fmt: skip


def keep_table(normal_rows: list[dict], kept: dict[str, bool], weight: str) -> dict:
    """The 2 x 2 table of normal or not against keeps or not, and its Fisher p-value.

    Args:
        normal_rows: The ``normality`` rows of one model.
        kept: Whether each pole keeps under NF4, by slug, for the poles that reproduce.
        weight: One of :data:`WEIGHTS`, or ``any`` for a pole with any non-normal weight.

    Returns:
        The counts and the p-value.
    """

    def non_normal(r: dict) -> bool:
        return (
            any(r["p"][w] < ALPHA for w in WEIGHTS) if weight == "any" else r["p"][weight] < ALPHA
        )

    cells = {"non_normal_fail": 0, "non_normal_keep": 0, "normal_fail": 0, "normal_keep": 0}
    for r in normal_rows:
        if r["slug"] in kept:
            group = "non_normal" if non_normal(r) else "normal"
            cells[f"{group}_{'keep' if kept[r['slug']] else 'fail'}"] += 1
    p = fisher(cells["non_normal_fail"], cells["non_normal_keep"], cells["normal_fail"],
               cells["normal_keep"])  # fmt: skip
    return {"weight": weight, **cells, "fisher_p": round(p, 4)}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@click.group()
def cli() -> None:
    """The analyses of the NF4 keep test."""


@cli.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def drift(results: Path) -> None:
    """Print the drift of the probes' peaks for each condition, one JSON line each.

    Args:
        results: The rows of ``scripts/nf4_keep.py``.
    """
    for row in drift_rows(_rows(results)):
        click.echo(json.dumps(row))


def instruct_overlap(rows: list[dict], instruct_rows: list[dict], model: str) -> dict:
    """The poles that flip under NF4 and under instruction tuning, on the poles both runs read.

    Args:
        rows: The rows of ``scripts/nf4_keep.py``.
        instruct_rows: The rows of ``scripts/random_delta_keep.py`` (#5154).
        model: The base model's name.

    Returns:
        The flips under each, how many flip under both, and the overlap that independent
        flips would give.
    """

    def poles(source: list[dict], condition: str) -> list[dict]:
        return [r for r in source if r["table"] == "pole" and r["model"] == model
                and r["condition"] == condition and r["base_passes"]]  # fmt: skip

    ours, theirs = poles(rows, "nf4"), poles(instruct_rows, "instruct")
    common = {r["slug"] for r in ours} & {r["slug"] for r in theirs}
    a = {r["slug"] for r in ours if not r["keeps"]} & common
    b = {r["slug"] for r in theirs if not r["keeps"]} & common
    expected = round(len(a) * len(b) / len(common), 2) if common else None
    return {"table": "instruct_overlap", "model": model, "poles": len(common),
            "nf4_flips": len(a), "instruct_flips": len(b), "both": len(a & b),
            "expected_both": expected}  # fmt: skip


@cli.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--instruct",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The rows of the instruct change (#5154), for the overlap of the flips.",
)
def paired(results: Path, instruct: Path | None) -> None:
    """Print NF4 against each seed of its random change, on the same poles, one line each.

    The rows are those of ``scripts/random_change_stats.py``: the flips under each change,
    the exact McNemar test, and the overlap of the flips with its hypergeometric test.

    Args:
        results: The rows of ``scripts/nf4_keep.py``.
        instruct: The rows of the instruct change, if any.
    """
    from random_change_stats import comparisons

    rows = _rows(results)
    for model in sorted({r["model"] for r in rows if r["table"] == "pole"}):
        for row in comparisons(rows, model, pairs=(("nf4", "random_nf4"),)):
            click.echo(json.dumps(row))
        if instruct:
            click.echo(json.dumps(instruct_overlap(rows, _rows(instruct), model)))


@cli.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def normality(results: Path, out: Path) -> None:
    """Test each pole's weights for normality, and whether non-normal poles keep less often.

    Args:
        results: The rows of ``scripts/nf4_keep.py``.
        out: The JSONL file of the normality rows and their tables.
    """
    from scipy.stats import shapiro

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openinterpretability.cli import _load_runner

    rows = _rows(results)
    lines = []
    for model in sorted({r["model"] for r in rows if r["table"] == "base"}):
        runner = _load_runner(model, "cpu", "float32", None)
        poles = [r for r in rows if r["table"] == "base" and r["model"] == model]
        normal_rows = []
        for r in poles:
            neuron = int(r["slug"].rsplit("/N", 1)[1].split("-")[0])
            vectors = weight_vectors(runner.model, r["layer"], neuron)
            p = {w: float(shapiro(v.detach().double().numpy()).pvalue) for w, v in vectors.items()}
            normal_rows.append({"table": "normality", "model": model, "slug": r["slug"], "p": p})
        nf4 = [r for r in rows if r["table"] == "pole" and r["model"] == model
               and r["condition"] == "nf4" and r["base_passes"]]  # fmt: skip
        kept = {r["slug"]: r["keeps"] for r in nf4}
        tables = [{"table": "normality_keep", "model": model, **keep_table(normal_rows, kept, w)}
                  for w in (*WEIGHTS, "any")]  # fmt: skip
        lines += normal_rows + tables
        for t in tables:
            click.echo(json.dumps(t))
    out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    cli()
