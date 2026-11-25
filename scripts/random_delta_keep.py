"""Test VERIFIED poles under instruction tuning and under random changes of its size.

Issue #5154 asks if a random change breaks as many explanations as instruction tuning.

The pair is a base model (``--base``) and its ``-Instruct`` model. Δ is the instruct
model's weight minus the base's, for each weight. Each condition adds a change to the base
model's weights in place, and copies the weights back afterwards:

* ``instruct``: all of Δ. This is the instruct model.
* ``instruct_mlp``: Δ on the MLP matrices only.
* ``random_mlp``: Gaussian noise on each MLP matrix, with the Frobenius norm of its own Δ.
* ``random_all``: Gaussian noise on each weight, with the Frobenius norm of its own Δ. This
  is the random floor of ``open-interpretability interference`` (#5230).

Each random condition runs once for each seed. Each VERIFIED pole of the base model is then
tested as ``scripts/base_instruct.py`` tests it (``interference.keep``). A pole keeps when
its probes pass both gates on the base model and on the changed model. The script writes
one row for each pole and condition, and a summary row for each condition.

    uv run python scripts/random_delta_keep.py --base Qwen/Qwen2.5-0.5B --device cuda \\
        --out papers/random-change-keep/qwen2.5-0.5b.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.interference.floors import _noise_like  # noqa: E402
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

# The names of the MLP matrices of a Qwen2 block end with these.
MLP_MATRICES = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
CONDITIONS = ("instruct", "instruct_mlp", "random_mlp", "random_all")
RANDOM = ("random_mlp", "random_all")


def is_mlp(name: str) -> bool:
    """True when the parameter ``name`` is one of a block's MLP matrices."""
    return name.endswith(MLP_MATRICES)


def deltas(base_model, other_model) -> dict[str, torch.Tensor]:
    """Each weight of ``other_model`` minus the same weight of ``base_model``, on the CPU."""
    theirs = dict(other_model.named_parameters())
    return {
        name: theirs[name].detach().float().cpu() - weight.detach().float().cpu()
        for name, weight in base_model.named_parameters()
    }


def change_for(condition: str, delta: dict[str, torch.Tensor], seed: int = 0) -> dict:
    """The change that ``condition`` adds to the base weights, keyed by parameter name.

    Args:
        condition: One of :data:`CONDITIONS`.
        delta: Δ for each weight, from :func:`deltas`.
        seed: The seed of the noise. Only the random conditions use it.

    Returns:
        ``{name: tensor}`` for each weight that the condition changes.

    Raises:
        ValueError: ``condition`` is not one of :data:`CONDITIONS`.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    every_weight = condition in ("instruct", "random_all")
    names = sorted(n for n in delta if every_weight or is_mlp(n))
    if condition in ("instruct", "instruct_mlp"):
        return {n: delta[n] for n in names}
    generator = torch.Generator().manual_seed(seed)
    return {n: _noise_like(delta[n], generator) for n in names}


@contextmanager
def changed(model, change: dict[str, torch.Tensor]):
    """``model`` with ``change`` added to its weights. The weights are copied back afterwards."""
    params = dict(model.named_parameters())
    saved = {n: params[n].detach().cpu().clone() for n in change}
    with torch.no_grad():
        for name, tensor in change.items():
            params[name].add_(tensor.to(params[name].device, params[name].dtype))
    try:
        yield model
    finally:
        with torch.no_grad():
            for name, weight in saved.items():
                params[name].copy_(weight.to(params[name].device))


def base_columns(runner, batches, records) -> dict[int, tuple[list[int], torch.Tensor]]:
    """``{layer: (neurons, columns)}``: the base activations of each layer's VERIFIED neurons."""
    by_layer: dict[int, set[int]] = defaultdict(set)
    for record in records:
        ident = record.submission.identifier
        by_layer[ident.layer].add(ident.neuron_idx)
    out = {}
    for layer, neurons in sorted(by_layer.items()):
        ordered = sorted(neurons)
        out[layer] = ordered, layer_columns(runner, batches, layer, ordered)
    return out


def pole_stats(runner, batches, records, base: dict) -> dict[str, dict]:
    """``{slug: {"ratio", "pearson"}}`` of the runner as it is now, against ``base``."""
    changed_cols = {
        layer: layer_columns(runner, batches, layer, neurons)
        for layer, (neurons, _) in base.items()
    }
    out = {}
    for record in records:
        ident = record.submission.identifier
        neurons, a = base[ident.layer]
        col = neurons.index(ident.neuron_idx)
        b = changed_cols[ident.layer]
        top = pole_max(a[:, col], ident.pole)
        ratio = pole_max(b[:, col], ident.pole) / top if top > 0 else None
        out[ident.slug] = {"ratio": ratio, "pearson": pearson(a[:, col], b[:, col])}
    return out


def summary(rows: list[dict], **fields) -> dict:
    """The keep share of one condition's pole rows, over the poles that reproduce."""
    reproduced = [r for r in rows if r["base_passes"]]
    kept = sum(r["keeps"] for r in reproduced)
    return {
        "table": "summary", **fields, "poles": len(rows),
        "reproduced_on_base": len(reproduced), "keeps": kept,
        "keep_share": round(kept / len(reproduced), 4) if reproduced else None,
        "keep_ci95": wilson(kept, len(reproduced)),
        "flipped": sorted(r["slug"] for r in reproduced if not r["keeps"]),
    }  # fmt: skip


def run_condition(runner, batches, records, base, on_base, change, fields) -> list[dict]:
    """The pole rows and the summary of one condition, with ``change`` added to the base."""
    rows = []
    with changed(runner.model, change):
        stats = pole_stats(runner, batches, records, base)
        for record in records:
            ident = record.submission.identifier
            s = stats[ident.slug]
            reading = readings(runner, record, s["ratio"] or 1.0)
            rows.append({
                "table": "pole", **fields, "slug": ident.slug, "layer": ident.layer,
                "pole": ident.pole, **s, "changed": reading,
                "base_passes": on_base[ident.slug]["passes"],
                "keeps": on_base[ident.slug]["passes"] and reading["passes"],
            })  # fmt: skip
    norm = float(torch.sqrt(sum((t.double() ** 2).sum() for t in change.values())))
    return [summary(rows, **fields, change_norm=round(norm, 4)), *rows]


@click.command()
@click.option("--base", "base_name", required=True, help="The base model, e.g. Qwen/Qwen2.5-0.5B.")
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--condition", "names", multiple=True, type=click.Choice(CONDITIONS))
@click.option("--mlp-seeds", default=5, show_default=True, help="Seeds of random_mlp.")
@click.option("--all-seeds", default=3, show_default=True, help="Seeds of random_all.")
def main(base_name, wiki_dir, out, device, names, mlp_seeds, all_seeds) -> None:
    """Test each VERIFIED pole under each condition, and write the rows to ``--out``."""
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, base_name)
    runner = _load_runner(base_name, device, "float32", None)
    other = _load_runner(f"{base_name}-Instruct", "cpu", "float32", None)
    delta = deltas(runner.model, other.model)
    del other
    batches = sample_windows(runner.tokenizer)
    base = base_columns(runner, batches, records)
    on_base = {r.submission.identifier.slug: readings(runner, r) for r in records}
    seeds = {"random_mlp": range(mlp_seeds), "random_all": range(all_seeds)}
    lines = []
    for name in names or CONDITIONS:
        for seed in seeds.get(name, [None]):
            fields = {"model": base_name, "condition": name, "seed": seed,
                      "sample_tokens": SAMPLE_TOKENS}  # fmt: skip
            change = change_for(name, delta, seed or 0)
            rows = run_condition(runner, batches, records, base, on_base, change, fields)
            lines += rows
            click.echo(json.dumps({k: v for k, v in rows[0].items() if k != "flipped"}), err=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    main()
