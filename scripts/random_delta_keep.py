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
from dataclasses import dataclass
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.harness.hooks import pole_strength  # noqa: E402
from openinterpretability.harness.units import NeuronUnit, submission_unit  # noqa: E402
from openinterpretability.interference.floors import _noise_like  # noqa: E402
from openinterpretability.interference.keep import (  # noqa: E402
    GATES,
    SAMPLE_TOKENS,
    layer_columns,
    pearson,
    pole_max,
    sample_windows,
    verified_records,
    wilson,
)

# The names of the MLP matrices of a Qwen2 block end with these.
MLP_MATRICES = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
CONDITIONS = ("instruct", "instruct_mlp", "random_mlp", "random_all")
RANDOM = ("random_mlp", "random_all")
# The most tokens, padding included, in one batched pass over the probe texts.
PROBE_TOKENS = 4096
# One-token inputs in one pass of the vocabulary sweep. Each pass records every layer.
FIRST_BATCH = 2048


def is_mlp(name: str) -> bool:
    """True when the parameter ``name`` is one of a block's MLP matrices.

    Args:
        name: A parameter's name, as ``named_parameters`` gives it.
    """
    return name.endswith(MLP_MATRICES)


def deltas(base_model, other_model) -> dict[str, torch.Tensor]:
    """Each weight of ``other_model`` minus the same weight of ``base_model``, on the CPU.

    Args:
        base_model: The base model.
        other_model: The model with the same parameter names, such as the instruct model.
    """
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
    """``model`` with ``change`` added to its weights. The weights are copied back afterwards.

    Args:
        model: The model to change in place.
        change: The tensor to add to each named parameter, from :func:`change_for`.
    """
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
    """``{layer: (neurons, columns)}``: the base activations of each layer's VERIFIED neurons.

    Args:
        runner: The base model.
        batches: The corpus sample, from ``interference.keep.sample_windows``.
        records: The VERIFIED records.
    """
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
    """``{slug: {"ratio", "pearson"}}`` of the runner as it is now, against ``base``.

    Args:
        runner: The model, as it is now.
        batches: The corpus sample that ``base`` was read on.
        records: The VERIFIED records.
        base: The base model's columns, from :func:`base_columns`.
    """
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


def probe_texts(record) -> list[str]:
    """The texts of the record's hard negatives and positive holdouts, which ``readings`` reads.

    Args:
        record: One VERIFIED record.
    """
    return [
        r.prompt for r in record.verdict.adversarial_test_results
        if r.kind in ("hard_negative", "positive_holdout")
    ]  # fmt: skip


def probe_activations(runner, records, max_tokens: int = PROBE_TOKENS) -> dict:
    """``{(text, layer, neuron): activations}`` for each probe text of each record.

    The texts run in batches of about ``max_tokens`` tokens, shortest first, and padded on
    the right. The model is causal, so a pad never changes the activation of an earlier
    token. Each pass records every layer that the batch's records need. Each text is
    tokenized alone, as ``readings`` tokenizes it.

    Args:
        runner: The model, as it is now.
        records: The VERIFIED records whose probes to read.
        max_tokens: The most tokens, padding included, in one pass.

    Returns:
        Each activation over the text's tokens, as a float32 CPU vector.
    """
    wanted: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for record in records:
        ident = record.submission.identifier
        for text in probe_texts(record):
            wanted[text].add((ident.layer, ident.neuron_idx))
    ids = {t: runner.encode(t)["input_ids"][0].cpu() for t in wanted}
    texts = sorted(wanted, key=lambda t: len(ids[t]))
    out = {}
    start = 0
    while start < len(texts):
        end = start + 1
        while end < len(texts) and (end - start + 1) * len(ids[texts[end]]) <= max_tokens:
            end += 1
        chunk = texts[start:end]
        width = len(ids[chunk[-1]])
        batch = torch.zeros(len(chunk), width, dtype=torch.long)
        for i, text in enumerate(chunk):
            batch[i, : len(ids[text])] = ids[text]
        layers = sorted({layer for t in chunk for layer, _ in wanted[t]})
        acts = runner.layer_activations({"input_ids": batch}, layers)
        for i, text in enumerate(chunk):
            for layer, neuron in wanted[text]:
                out[text, layer, neuron] = acts[layer][i, : len(ids[text]), neuron].float().cpu()
        start = end
    return out


def first_position_activations(runner, records, batch_size: int = FIRST_BATCH) -> dict:
    """``{(layer, neuron): activations}`` at the first position of every vocabulary token.

    This is ``harness.first_position.strengths_of`` for every record's neuron in one sweep
    of the vocabulary. Each pass records every layer that the records need.

    Args:
        runner: The model, as it is now.
        records: The VERIFIED records whose neurons to read.
        batch_size: The one-token inputs in one pass.

    Returns:
        Each neuron's raw activation on each token of ``first_position.vocabulary``.
    """
    from openinterpretability.harness.first_position import vocabulary

    by_layer: dict[int, set[int]] = defaultdict(set)
    for record in records:
        ident = record.submission.identifier
        by_layer[ident.layer].add(ident.neuron_idx)
    neurons = {layer: sorted(n) for layer, n in by_layer.items()}
    vocab, parts = vocabulary(runner), defaultdict(list)
    for start in range(0, len(vocab), batch_size):
        ids = torch.tensor(vocab[start : start + batch_size], device=runner.device).unsqueeze(1)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        acts = runner.layer_activations(inputs, sorted(neurons))
        for layer, ns in neurons.items():
            parts[layer].append(acts[layer][:, 0, ns].float().cpu())
    out = {}
    for layer, ns in neurons.items():
        column = torch.cat(parts[layer])
        for i, neuron in enumerate(ns):
            out[layer, neuron] = column[:, i]
    return out


@dataclass(frozen=True)
class FirstPositions:
    """A pole's first-position strengths, in the form ``scan.first_position_share`` reads."""

    raw: torch.Tensor
    pole: str

    def first_position(self) -> torch.Tensor:
        """The pole's strength at the first position, on every token of the vocabulary."""
        return pole_strength(self.raw, self.pole)


def settle_cached(record, runner, policy, firsts: dict):
    """``settle_first_token_floor``, with the first positions read from ``firsts``.

    Args:
        record: One VERIFIED record.
        runner: The model, as it was when ``firsts`` was read.
        policy: The record's review policy.
        firsts: The first positions, from :func:`first_position_activations`.

    Returns:
        ``(unit, scale)``, as ``settle_first_token_floor`` returns them.
    """
    from openinterpretability.agents.review_measurements import settle_first_token_floor
    from openinterpretability.harness.scan import has_first_token_floor

    submission = record.submission
    if submission.first_token_floor is not None or not submission.max_activation:
        _, unit, scale = settle_first_token_floor(submission, runner, policy)
        return unit, scale
    ident, scale = submission.identifier, submission.max_activation
    reader = FirstPositions(firsts[ident.layer, ident.neuron_idx], ident.pole)
    floor = has_first_token_floor(runner, reader, scale, policy)
    return submission_unit(submission.model_copy(update={"first_token_floor": floor})), scale


@dataclass(frozen=True)
class CachedUnit:
    """A neuron pole whose strengths come from :func:`probe_activations`, not the model."""

    unit: NeuronUnit
    cache: dict

    @property
    def skip_first(self) -> bool:
        """The pole's first-token floor, as its unit has it."""
        return self.unit.skip_first

    def strengths(self, runner, text: str) -> torch.Tensor:  # noqa: ARG002
        """Per-token pole strength ``[1, seq]`` of ``text``, from the cache.

        Args:
            runner: Not read. ``Unit.strengths`` takes it.
            text: One probe text that :func:`probe_activations` read.
        """
        ident = self.unit.ident
        act = self.cache[text, ident.layer, ident.neuron_idx]
        return pole_strength(act, ident.pole).unsqueeze(0)


def cached_readings(runner, record, cache: dict, firsts: dict, scale_factor: float = 1.0) -> dict:
    """``interference.keep.readings``, with the model's readings taken from the caches.

    The scale, the threshold, the probes and the gates are the ones ``readings`` uses.

    Args:
        runner: The model, as it was when the caches were read.
        record: One VERIFIED record.
        cache: The probes' activations, from :func:`probe_activations`.
        firsts: The first positions, from :func:`first_position_activations`.
        scale_factor: Multiplies the record's scale, as in ``readings``.
    """
    from openinterpretability.agents.review_measurements import measure_probes
    from openinterpretability.agents.reviewer import AdversarialProbe, review_metrics
    from openinterpretability.core.schemas import ReviewPolicy

    policy = record.verdict.policy or ReviewPolicy()
    probes = [
        AdversarialProbe(text=r.prompt, expected_active=r.expected_active, rationale="")
        for r in record.verdict.adversarial_test_results
        if r.kind in ("hard_negative", "positive_holdout")
    ]
    unit, scale = settle_cached(record, runner, policy, firsts)
    scale *= scale_factor
    threshold = policy.activation_threshold * scale
    results, _, _ = measure_probes(runner, CachedUnit(unit, cache), probes, scale, threshold)
    metrics = review_metrics(results, policy, independent=True)
    return {
        "specificity": metrics.specificity, "sensitivity": metrics.sensitivity,
        "balanced_accuracy": metrics.balanced_accuracy,
        "passes": all(metrics.gates[g] for g in GATES),
        "failed": [g for g in GATES if not metrics.gates[g]],
    }  # fmt: skip


def summary(rows: list[dict], **fields) -> dict:
    """The keep share of one condition's pole rows, over the poles that reproduce.

    Args:
        rows: The condition's pole rows, from :func:`run_condition`.
        **fields: The condition's fields, such as its name and seed, copied into the row.
    """
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
    """The pole rows and the summary of one condition, with ``change`` added to the base.

    Args:
        runner: The base model, which the condition changes and then puts back.
        batches: The corpus sample, from ``interference.keep.sample_windows``.
        records: The VERIFIED records.
        base: The base model's columns, from :func:`base_columns`.
        on_base: Each pole's readings on the base model, by slug.
        change: The change to add, from :func:`change_for`.
        fields: The condition's fields, copied into each row.

    Returns:
        The summary row, then one row for each pole.
    """
    rows = []
    with changed(runner.model, change):
        stats = pole_stats(runner, batches, records, base)
        cache = probe_activations(runner, records)
        firsts = first_position_activations(runner, records)
        for record in records:
            ident = record.submission.identifier
            s = stats[ident.slug]
            reading = cached_readings(runner, record, cache, firsts, s["ratio"] or 1.0)
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
    """Test each VERIFIED pole under each condition, and write the rows to ``--out``.

    Args:
        base_name: The base model. Its instruct model is ``<base_name>-Instruct``.
        wiki_dir: The registry that holds the VERIFIED records.
        out: The JSONL file to write. It is written again after each condition.
        device: The device of the base model. The instruct model loads on the CPU.
        names: The conditions to run. With none, every condition runs.
        mlp_seeds: The number of seeds of ``random_mlp``.
        all_seeds: The number of seeds of ``random_all``.
    """
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, base_name)
    runner = _load_runner(base_name, device, "float32", None)
    other = _load_runner(f"{base_name}-Instruct", "cpu", "float32", None)
    delta = deltas(runner.model, other.model)
    del other
    batches = sample_windows(runner.tokenizer)
    base = base_columns(runner, batches, records)
    cache, firsts = probe_activations(runner, records), first_position_activations(runner, records)
    on_base = {
        r.submission.identifier.slug: cached_readings(runner, r, cache, firsts) for r in records
    }
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
