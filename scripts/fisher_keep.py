"""Does a pole's Fisher importance predict whether its explanation keeps (#5118)?

The outcomes are the stored rows of ``docs/analyses/base-instruct-0.5b.jsonl``. Each row
says whether a VERIFIED pole of Qwen2.5-0.5B keeps its explanation on the instruct model.
The ``-chat`` rows give the same outcome with the probes inside the chat template.

The script reads the diagonal Fisher of every MLP neuron of the base model
(``interference/fisher.py``). The sample is part of ``corpus.jsonl``. Each pole gets the
percentile of its neuron's Fisher within its layer. Then the script tests each percentile
against each outcome. The null shuffles the outcomes within each layer. The script changes
no weights and writes no record.

    uv run python scripts/fisher_keep.py --device cpu --cache fisher.pt \\
        --out papers/fisher-keep/results.jsonl
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.interference.fisher import (  # noqa: E402
    LABELS,
    PERMUTATIONS,
    SEED,
    SIDES,
    activation_rms,
    auc,
    diagonal_fisher,
    fisher_windows,
    layer_percentile,
    partial_spearman,
    spearman,
    stratified_null,
)
from openinterpretability.interference.keep import GATES, sample_windows  # noqa: E402

BASE = "Qwen/Qwen2.5-0.5B"
PLAIN = Path("docs/analyses/base-instruct-0.5b.jsonl")
CHAT = Path("docs/analyses/base-instruct-0.5b-chat.jsonl")
FLOORS = {"specificity": 0.80, "sensitivity": 0.60}  # the gates that base-instruct.md reads
WEIGHTS = (*SIDES, "total")  # the input weights, the output weights, and both


def neuron_of(slug: str) -> int:
    """The neuron index in a slug such as ``Qwen--Qwen2.5-0.5B/L0/N689-negative``.

    Args:
        slug: The pole's slug.

    Returns:
        The neuron's index in its layer.

    Raises:
        ValueError: The slug names no neuron.
    """
    match = re.search(r"/N(\d+)-", slug)
    if not match:
        raise ValueError(f"no neuron index in {slug!r}")
    return int(match.group(1))


def margin(base: dict) -> float:
    """How far the base readings sit above the nearer gate.

    Args:
        base: The base model's readings of a row, with each gate's value.

    Returns:
        The smaller of the two distances above :data:`FLOORS`.
    """
    return min(base[g] - FLOORS[g] for g in GATES)


def load_poles(plain: Path, chat: Path) -> list[dict]:
    """The pole rows of ``plain``, each with its chat outcome from ``chat`` when it has one.

    Args:
        plain: The base-instruct rows with plain probes.
        chat: The base-instruct rows with probes inside the chat template.

    Returns:
        One dict for each pole that reproduces on the base model.
    """

    def rows(path):
        """The ``pole`` rows of one JSON Lines file.

        Args:
            path: The file.

        Returns:
            The rows, as dicts.
        """
        return [r for r in map(json.loads, path.read_text().splitlines()) if r["table"] == "pole"]

    chat_keeps = {r["slug"]: r["keeps"] for r in rows(chat)}
    out = []
    for r in rows(plain):
        if not r["base"]["passes"]:
            continue  # a pole that doesn't reproduce on the base model has no outcome
        out.append({
            "slug": r["slug"], "layer": r["layer"], "neuron": neuron_of(r["slug"]),
            "pole": r["pole"], "depth": r["depth"], "keeps": r["keeps"],
            "keeps_chat": chat_keeps.get(r["slug"]), "pearson": r["pearson"],
            "ratio": r["ratio"], "margin": round(margin(r["base"]), 6),
        })  # fmt: skip
    return out


def add_fisher(poles: list[dict], fisher: dict, rms: torch.Tensor) -> None:
    """Put each pole's Fisher, its activation RMS and their within-layer percentiles on its row.

    Args:
        poles: The rows of :func:`load_poles`. They change in place.
        fisher: The Fisher of :func:`diagonal_fisher`.
        rms: The activation RMS of :func:`activation_rms`.
    """
    for p in poles:
        p["rms"] = float(rms[p["layer"], p["neuron"]])
        p["fisher"] = {}
        p["percentile"] = {"rms": round(layer_percentile(rms, p["layer"], p["neuron"]), 6)}
        for label in LABELS:
            values = {**fisher[label], "total": fisher[label]["input"] + fisher[label]["output"]}
            p["fisher"][label] = {w: float(values[w][p["layer"], p["neuron"]]) for w in WEIGHTS}
            p["percentile"][label] = {
                w: round(layer_percentile(values[w], p["layer"], p["neuron"]), 6) for w in WEIGHTS
            }


def _row(name, kind, label, w, units, res) -> dict:
    """One ``test`` row.

    Args:
        name: The outcome.
        kind: The statistic: ``auc``, ``spearman`` or ``partial_spearman``.
        label: The Fisher's kind of label, or ``rms``.
        w: The weights: ``input``, ``output``, ``total`` or ``activation``.
        units: The poles in the test.
        res: The result of :func:`stratified_null`.

    Returns:
        The row, with the numbers rounded.
    """
    return {"table": "test", "outcome": name, "statistic": kind, "fisher": label,
            "weights": w, "n": len(units), "observed": round(res["observed"], 4),
            "null_mean": round(res["null_mean"], 4), "p": round(res["p"], 4)}  # fmt: skip


OUTCOMES = {  # each outcome: its statistic, and how a pole's row gives it
    "keeps": ("auc", lambda p: p["keeps"]),
    "keeps_chat": ("auc", lambda p: p["keeps_chat"]),
    "one_minus_r": ("spearman", lambda p: 1 - p["pearson"]),
    "abs_log_ratio": ("spearman", lambda p: abs(math.log(p["ratio"]))),
    "margin": ("spearman", lambda p: p["margin"]),
}
PAIRS = [(label, w) for label in LABELS for w in WEIGHTS] + [("rms", "activation")]


def statistic(kind: str, scores: list[float], control: list[float] | None = None):
    """The statistic of ``scores`` against outcomes, as a function of the outcomes.

    Args:
        kind: ``auc``, ``spearman`` or ``partial_spearman``.
        scores: The percentiles, one for each pole.
        control: The RMS percentiles, for ``partial_spearman`` only.

    Returns:
        A function that takes the outcomes, in the order of ``scores``, and gives the
        statistic. A Spearman ρ that is undefined gives 0.
    """
    if kind == "auc":
        return lambda ys: auc(scores, ys)
    if kind == "spearman":
        return lambda ys: spearman(scores, ys) or 0.0
    return lambda ys: partial_spearman(scores, ys, control) or 0.0


def tests(poles: list[dict]) -> list[dict]:
    """Each percentile against each outcome, with its within-layer null.

    The percentiles are each Fisher's and the activation RMS's (``fisher`` ``"rms"``).
    For the Spearman outcomes, a ``partial_spearman`` row also takes the RMS out.

    Args:
        poles: The rows of :func:`load_poles`, after :func:`add_fisher`.

    Returns:
        One ``test`` row for each test. The keep test of the total model Fisher also
        holds its null values, in shuffle order.
    """
    out = []
    for name, (kind, read) in OUTCOMES.items():
        units = [p for p in poles if read(p) is not None]
        ys, strata = [read(p) for p in units], [p["layer"] for p in units]
        control = [p["percentile"]["rms"] for p in units]
        for label, w in PAIRS:
            scores = [p["percentile"][label] if label == "rms" else p["percentile"][label][w]
                      for p in units]  # fmt: skip
            res = stratified_null(statistic(kind, scores), ys, strata)
            row = _row(name, kind, label, w, units, res)
            if name == "keeps" and label == "model" and w == "total":
                row["null"] = [round(v, 4) for v in res["null"]]  # in shuffle order, for the figure
            out.append(row)
            if kind == "spearman" and label != "rms":
                res = stratified_null(statistic("partial_spearman", scores, control), ys, strata)
                out.append(_row(name, "partial_spearman", label, w, units, res))
    return out


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option(
    "--cache",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Save the Fisher here, and read it from here if it exists.",
)
@click.option("--device", default="cpu", show_default=True)
@click.option("--corpus", default="corpus.jsonl", show_default=True)
def main(out: Path, cache: Path | None, device: str, corpus: str) -> None:
    """Read the Fisher of the base model, test it against the outcomes, and write the rows.

    Args:
        out: The JSON Lines file to write.
        cache: The file that holds the Fisher and the RMS, or ``None`` for no cache.
        device: The device of the model.
        corpus: The corpus of the sample.
    """
    from openinterpretability.cli import _load_runner

    poles = load_poles(PLAIN, CHAT)
    saved = torch.load(cache) if cache and cache.exists() else {}
    if "fisher" not in saved or "rms" not in saved:
        runner = _load_runner(BASE, device, "float32", None)
        rows = fisher_windows(sample_windows(runner.tokenizer, corpus=corpus))
        if "fisher" not in saved:
            saved["fisher"], saved["rows"] = diagonal_fisher(runner, rows), len(rows)
        saved["rms"] = activation_rms(runner, rows)
        if cache:
            torch.save(saved, cache)
    fisher, rms, n_rows = saved["fisher"], saved["rms"], saved["rows"]
    add_fisher(poles, fisher, rms)
    layers = sorted({p["layer"] for p in poles})
    by_layer = [{
        "table": "layer", "layer": layer,
        "median_fisher": {k: float((v["input"][layer] + v["output"][layer]).median())
                          for k, v in fisher.items()},
        "poles": sum(p["layer"] == layer for p in poles),
        "keeps": sum(p["layer"] == layer and p["keeps"] for p in poles),
    } for layer in layers]  # fmt: skip
    results = tests(poles)
    summary = {"table": "summary", "base": BASE, "poles": len(poles),
               "keeps": sum(p["keeps"] for p in poles), "fisher_rows": n_rows,
               "permutations": PERMUTATIONS, "seed": SEED}  # fmt: skip
    rows_out = [summary, *results, *by_layer, *({"table": "pole", **p} for p in poles)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows_out))
    for r in results:
        click.echo(f"{r['outcome']:14} {r['fisher']:5} {r['weights']:6} {r['statistic']:8} "
                   f"{r['observed']:+.3f} null {r['null_mean']:+.3f} p={r['p']:.4f}")  # fmt: skip


if __name__ == "__main__":
    main()
