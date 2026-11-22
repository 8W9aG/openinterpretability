"""Which VERIFIED explanations keep when a neuron's input weights are pruned (#5122)?

For each neuron pole of ``--model`` that stands VERIFIED, the script zeroes a share of the
neuron's input weights (``interference/pruning.py``). It does this for each level and each
method, magnitude and random. Then it runs the keep test of ``docs/analyses/base-instruct.md``
(``interference/keep.py``). It measures the record's own probes with the record's scale
times a ratio. The ratio is the pole's maximum after pruning over its maximum before, on a
sample of ``corpus.jsonl``.
A pole keeps when the specificity and sensitivity gates pass. The script also gives the
Pearson correlation of the pole's activations before and after pruning. It writes no record,
and it saves no weights.

    uv run python scripts/pruning_keep.py --device cpu --out papers/pruning-keep/results.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.interference.keep import (  # noqa: E402
    SAMPLE_TOKENS,
    layer_columns,
    pearson,
    pole_max,
    readings,
    sample_windows,
    verified_records,
    wilson,
)
from openinterpretability.interference.pruning import LEVELS, METHODS, SEED, pruned  # noqa: E402

MODEL = "EleutherAI/pythia-70m"


def by_layer(records: list) -> dict[int, list]:
    """The records grouped by the layer of their pole.

    Args:
        records: The VERIFIED records.

    Returns:
        ``{layer: [record, ...]}``, in layer order.
    """
    out: dict[int, list] = defaultdict(list)
    for record in records:
        out[record.submission.identifier.layer].append(record)
    return dict(sorted(out.items()))


def measure(runner, batches, records: list, variant: tuple | None, base: dict) -> list[dict]:
    """One row for each pole of one layer, read with ``variant`` applied.

    Args:
        runner: The model's ``NeuronHookRunner``.
        batches: The corpus sample, from :func:`sample_windows`.
        records: The VERIFIED records of the layer's poles.
        variant: ``(fraction, method)`` to prune, or None for the unpruned model.
        base: The unpruned activation columns of each layer, by layer, filled on the
            unpruned pass and read on the others.

    Returns:
        The rows.
    """
    layer = records[0].submission.identifier.layer
    neurons = sorted({r.submission.identifier.neuron_idx for r in records})
    rows = []
    if variant is None:  # the unpruned pass: the other variants compare with its columns
        base[layer] = layer_columns(runner, batches, layer, neurons)
        context, fraction, method = nullcontext(), 0.0, "none"
    else:
        fraction, method = variant
        context = pruned(runner, layer, neurons, fraction, method)
    with context:
        cols = layer_columns(runner, batches, layer, neurons) if variant else base[layer]
        for record in records:
            ident = record.submission.identifier
            col = neurons.index(ident.neuron_idx)
            top = pole_max(base[layer][:, col], ident.pole)
            ratio = pole_max(cols[:, col], ident.pole) / top if top > 0 else None
            read = readings(runner, record, ratio or 1.0)
            rows.append({
                "table": "pole", "slug": ident.slug, "layer": ident.layer, "pole": ident.pole,
                "fraction": fraction, "method": method, "ratio": ratio,
                "pearson": pearson(base[layer][:, col], cols[:, col]),
                "specificity": read["specificity"], "sensitivity": read["sensitivity"],
                "failed": read["failed"], "keeps": read["passes"],
            })  # fmt: skip
    return rows


def summary(rows: list[dict]) -> list[dict]:
    """One row for each level and method: how many poles keep, of those that reproduce.

    Args:
        rows: The ``pole`` rows of every variant.

    Returns:
        The ``level`` rows. A pole counts only when it passes on the unpruned model.
    """
    reproduced = {r["slug"] for r in rows if r["method"] == "none" and r["keeps"]}
    out = []
    for fraction in LEVELS:
        for method in METHODS:
            got = [r for r in rows if r["fraction"] == fraction and r["method"] == method
                   and r["slug"] in reproduced]  # fmt: skip
            kept = sum(r["keeps"] for r in got)
            rs = sorted(r["pearson"] for r in got if r["pearson"] is not None)
            out.append({
                "table": "level", "fraction": fraction, "method": method, "poles": len(got),
                "keeps": kept, "keep_share": round(kept / len(got), 4) if got else None,
                "keep_ci95": wilson(kept, len(got)),
                "median_pearson": rs[len(rs) // 2] if rs else None,
            })  # fmt: skip
    return out


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--model", default=MODEL, show_default=True)
@click.option("--corpus", default="corpus.jsonl", show_default=True)
def main(wiki_dir: Path, out: Path, device: str, model: str, corpus: str) -> None:
    """Prune each VERIFIED pole's input weights at each level, and write the keep rows.

    Args:
        wiki_dir: The registry's wiki.
        out: The JSON Lines file to write.
        device: The device of the model.
        model: The model, by its Hugging Face name.
        corpus: The corpus of the sample.
    """
    from openinterpretability.cli import _load_runner

    runner = _load_runner(model, device, "float32", None)
    batches = sample_windows(runner.tokenizer, corpus=corpus)
    variants = [None, *((f, m) for f in LEVELS for m in METHODS)]
    rows, base = [], {}
    for layer, records in by_layer(verified_records(wiki_dir, model)).items():
        for variant in variants:  # the unpruned pass first: the others read its columns
            rows += measure(runner, batches, records, variant, base)
        click.echo(f"layer {layer}: {len(records)} poles", err=True)
    levels = summary(rows)
    head = {"table": "summary", "model": model, "poles": len({r["slug"] for r in rows}),
            "reproduced": sum(r["keeps"] for r in rows if r["method"] == "none"),
            "sample_tokens": SAMPLE_TOKENS, "levels": list(LEVELS), "seed": SEED}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in [head, *levels, *rows]))
    for r in levels:
        click.echo(f"{r['fraction']:.2f} {r['method']:9} {r['keeps']}/{r['poles']} "
                   f"r={r['median_pearson']}")  # fmt: skip


if __name__ == "__main__":
    main()
