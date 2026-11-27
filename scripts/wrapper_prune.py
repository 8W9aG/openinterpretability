"""Prune a few instruct-model neurons by gradient × weight, and test the failed poles again (#5163).

Jain et al. (2024) find that fine-tuning often adds a small wrapper on top of a pretrained
capability. Pruning a few neurons by gradient × weight on the old task's loss brings the
capability back. This script asks the same of Qwen2.5-0.5B's instruct model.
The capability is each VERIFIED explanation that instruction tuning breaks.

The instruct model is the base model with Δ added in place (``random_delta_keep.changed``).
On the corpus sample of the keep test, the script backpropagates the mean token
KL(base ‖ instruct). A neuron's score is the sum of gradient × weight over its ``down_proj``
column. That is the first-order drop in the KL when the column is zeroed. Pruning zeroes the
column, so the neuron's write goes and its own activation stays. The conditions are these:

* ``none``: the instruct model.
* ``top``: the K neurons with the highest score.
* ``random``: K random neurons with the same count in each layer as ``top``, for each seed.

Each VERIFIED pole is then tested as ``scripts/base_instruct.py`` tests it, base against the
pruned instruct model. The KL to the base model is read again on held-out windows.

Two variants change one part each (#7071):

* ``--unit input`` prunes a neuron's input rows (``gate_proj`` and ``up_proj``) in place of
  its write column, so its activation is 0. The score sums gradient × weight over both rows.
* ``--loss failed`` takes the KL on the probe texts of the poles that fail on the instruct
  model, in place of the corpus sample. Only half of those poles, drawn with seed 0, give
  texts to the score. The other half is the held-out check.

    uv run python scripts/wrapper_prune.py --base Qwen/Qwen2.5-0.5B --device cuda \\
        --out papers/wrapper-prune/results.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_delta_keep import (  # noqa: E402
    base_columns,
    base_rows,
    cached_readings,
    changed,
    deltas,
    first_position_activations,
    probe_activations,
    probe_texts,
    run_condition,
)

from openinterpretability.interference.keep import (  # noqa: E402
    SAMPLE_TOKENS,
    sample_windows,
    verified_records,
)

KS = (5, 15, 50, 150, 500)  # the neurons pruned
DOWN = "mlp.down_proj.weight"
INPUTS = ("mlp.gate_proj.weight", "mlp.up_proj.weight")  # a neuron's input rows
UNITS = ("write", "input")  # what a prune zeroes: the down column, or the input rows
LOSSES = ("base", "failed")  # the old task: the corpus sample, or the failed poles' texts
SPLIT_SEED = 0  # the draw of the failed poles whose texts the score reads


def down_name(layer: int) -> str:
    """The parameter name of layer ``layer``'s ``down_proj`` in a Qwen2 model.

    Args:
        layer: The layer.
    """
    return f"model.layers.{layer}.{DOWN}"


def input_names(layer: int) -> tuple[str, ...]:
    """The parameter names of layer ``layer``'s ``gate_proj`` and ``up_proj`` in a Qwen2 model.

    Args:
        layer: The layer.
    """
    return tuple(f"model.layers.{layer}.{name}" for name in INPUTS)


def token_kl(base_logp: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """The mean over tokens of KL(base ‖ model), from base log-probabilities and model logits.

    Args:
        base_logp: The base model's log-probabilities, ``[tokens, vocab]``.
        logits: The model's logits, ``[tokens, vocab]``.

    Returns:
        A scalar.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    base = base_logp.float()
    return (base.exp() * (base - logp)).sum(-1).mean()


def base_logprobs(model, windows: list[torch.Tensor], device) -> list[torch.Tensor]:
    """The model's next-token log-probabilities for each window, on the CPU in float16.

    Args:
        model: The base model.
        windows: Rows of token ids, one ``[1, n]`` tensor each.
        device: The model's device.

    Returns:
        One ``[n, vocab]`` tensor for each window.
    """
    out = []
    with torch.no_grad():
        for ids in windows:
            logits = model(ids.to(device)).logits[0]
            out.append(torch.log_softmax(logits.float(), dim=-1).half().cpu())
    return out


def mean_kl(model, windows, base_lp, device) -> float:
    """The mean token KL(base ‖ model) over ``windows``.

    Args:
        model: The model, as it is now.
        windows: Rows of token ids.
        base_lp: The base model's log-probabilities for the same rows.
        device: The model's device.
    """
    total = 0.0
    with torch.no_grad():
        for ids, lp in zip(windows, base_lp, strict=True):
            total += float(token_kl(lp.to(device), model(ids.to(device)).logits[0]))
    return total / len(windows)


def scores(model, windows, base_lp, device, unit: str = "write") -> dict[int, torch.Tensor]:
    """Each layer's neuron scores: gradient × weight of the KL, summed over the pruned weights.

    Args:
        model: The instruct model.
        windows: Rows of token ids.
        base_lp: The base model's log-probabilities for the same rows.
        device: The model's device.
        unit: ``write`` sums over the neuron's down column. ``input`` sums over its rows of
            ``gate_proj`` and ``up_proj``.

    Returns:
        ``{layer: [neurons]}`` on the CPU. A higher score is a larger first-order drop in the
        KL when the neuron's weights are zeroed.
    """
    if unit == "write":
        weights = {i: [layer.mlp.down_proj.weight] for i, layer in enumerate(model.model.layers)}
    else:
        weights = {i: [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight]
                   for i, layer in enumerate(model.model.layers)}  # fmt: skip
    saved = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for ws in weights.values():
        for w in ws:
            w.requires_grad_(True)
            w.grad = None
    try:
        for ids, lp in zip(windows, base_lp, strict=True):
            # The mean over windows, so that each window counts once.
            loss = token_kl(lp.to(device), model(ids.to(device)).logits[0]) / len(windows)
            loss.backward()
        # Zeroing a weight changes the KL by about -(gradient × weight). A down column is
        # dimension 1 of down_proj, and an input row is dimension 0 of gate_proj and up_proj.
        dim = 0 if unit == "write" else 1
        return {i: sum((w.grad * w.detach()).sum(dim=dim) for w in ws).cpu()
                for i, ws in weights.items()}  # fmt: skip
    finally:
        for name, p in model.named_parameters():
            p.grad = None
            p.requires_grad_(saved[name])


def top_neurons(layer_scores: dict[int, torch.Tensor], k: int) -> list[tuple[int, int]]:
    """The ``k`` neurons with the highest score, over every layer, as ``(layer, neuron)``.

    Args:
        layer_scores: Each layer's neuron scores, from :func:`scores`.
        k: How many.
    """
    flat = [(float(s), layer, n) for layer, row in layer_scores.items()
            for n, s in enumerate(row.tolist())]  # fmt: skip
    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    return sorted((layer, n) for _, layer, n in flat[:k])


def random_like(chosen: list[tuple[int, int]], widths: dict[int, int], seed: int) -> list:
    """Random neurons with the same count in each layer as ``chosen``, none of them chosen.

    Args:
        chosen: The neurons to match, as ``(layer, neuron)``.
        widths: Each layer's number of neurons.
        seed: The seed of the draw.

    Returns:
        The drawn neurons, as ``(layer, neuron)``.
    """
    generator = torch.Generator().manual_seed(seed)
    out = []
    for layer in sorted({layer for layer, _ in chosen}):
        taken = {n for lay, n in chosen if lay == layer}
        count = len(taken)
        pool = torch.tensor([n for n in range(widths[layer]) if n not in taken])
        picks = torch.randperm(len(pool), generator=generator)[:count]
        out += [(layer, int(pool[i])) for i in sorted(picks.tolist())]
    return out


def pruned_change(delta: dict, base_model, neurons: list[tuple[int, int]],
                  unit: str = "write") -> dict:  # fmt: skip
    """Δ with each pruned neuron's weights set so that the instruct weights become zero.

    Args:
        delta: Δ for each weight, from ``random_delta_keep.deltas``.
        base_model: The base model.
        neurons: The neurons to prune, as ``(layer, neuron)``.
        unit: ``write`` zeroes the neuron's down column. ``input`` zeroes its rows of
            ``gate_proj`` and ``up_proj``.

    Returns:
        The change to add to the base weights: the pruned instruct model.
    """
    params = dict(base_model.named_parameters())
    change = dict(delta)
    for layer in {layer for layer, _ in neurons}:
        picked = [n for lay, n in neurons if lay == layer]
        if unit == "write":
            name = down_name(layer)
            column = change[name].clone()
            column[:, picked] = -params[name].detach().float().cpu()[:, picked]
            change[name] = column
            continue
        for name in input_names(layer):
            rows = change[name].clone()
            rows[picked, :] = -params[name].detach().float().cpu()[picked, :]
            change[name] = rows
    return change


def split_failed(slugs: list[str], seed: int = SPLIT_SEED) -> tuple[list[str], list[str]]:
    """The failed poles in two halves: those whose texts the score reads, and the held-out rest.

    Args:
        slugs: The poles that fail on the instruct model.
        seed: The seed of the draw.

    Returns:
        ``(scored, held)``, each sorted. ``scored`` has the larger half when the count is odd.
    """
    order = sorted(slugs)
    picks = torch.randperm(len(order), generator=torch.Generator().manual_seed(seed)).tolist()
    half = (len(order) + 1) // 2
    return sorted(order[i] for i in picks[:half]), sorted(order[i] for i in picks[half:])


def text_windows(tokenizer, records) -> list[torch.Tensor]:
    """The probe texts of ``records``, each as one ``[1, n]`` row of token ids.

    Args:
        tokenizer: The model's tokenizer.
        records: The records whose hard negatives and positive holdouts to read.
    """
    return [torch.tensor([tokenizer(text)["input_ids"]])
            for record in records for text in probe_texts(record)]  # fmt: skip


def scoring_windows(tokenizer, records, none_rows, windows, loss) -> tuple[list, dict | None]:
    """The rows that the score reads, and the split of the failed poles when there is one.

    Args:
        tokenizer: The model's tokenizer.
        records: The VERIFIED records.
        none_rows: The rows of the ``none`` condition, which name the failed poles.
        windows: The corpus sample's rows, which the ``base`` loss reads.
        loss: One of ``LOSSES``.

    Returns:
        ``(rows, split)``. ``split`` is ``None`` for the ``base`` loss. For ``failed`` it is
        ``{"seed", "scored", "held"}``, and the rows are the scored half's probe texts.
    """
    if loss == "base":
        return windows, None
    scored, held = split_failed(failed_slugs(none_rows))
    chosen = set(scored)
    picked = [r for r in records if r.submission.identifier.slug in chosen]
    return text_windows(tokenizer, picked), {"seed": SPLIT_SEED, "scored": scored, "held": held}


def failed_slugs(rows: list[dict]) -> list[str]:
    """The poles that pass on the base model and fail on the instruct model.

    Args:
        rows: The pole rows of the ``none`` condition, from ``run_condition``.
    """
    return [r["slug"] for r in rows if r["table"] == "pole" and r["base_passes"]
            and not r["keeps"]]  # fmt: skip


def prune_plan(layer_scores: dict[int, torch.Tensor], seeds: int) -> list[tuple]:
    """Each condition to run: ``(condition, K, seed, neurons)``, the instruct model first.

    Args:
        layer_scores: Each layer's neuron scores, from :func:`scores`.
        seeds: The number of seeds of the random prune at each K.
    """
    widths = {layer: len(v) for layer, v in layer_scores.items()}
    plan = [("none", 0, None, [])]
    for k in KS:
        chosen = top_neurons(layer_scores, k)
        plan += [("top", k, None, chosen)]
        plan += [("random", k, s, random_like(chosen, widths, s)) for s in range(seeds)]
    return plan


@click.command()
@click.option("--base", "base_name", required=True, help="The base model, e.g. Qwen/Qwen2.5-0.5B.")
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--seeds", default=3, show_default=True, help="Seeds of the random prune.")
@click.option("--unit", type=click.Choice(UNITS), default="write", show_default=True,
              help="Prune the down column (write) or the input rows (input).")  # fmt: skip
@click.option("--loss", type=click.Choice(LOSSES), default="base", show_default=True,
              help="Score on the corpus (base) or the failed poles' texts (failed).")  # fmt: skip
def main(base_name, wiki_dir, out, device, seeds, unit, loss) -> None:  # long-function: one run
    # loads both models once, then runs every condition in order.
    """Prune the top-scored neurons and random ones, and test each VERIFIED pole.

    Args:
        base_name: The base model. Its instruct model is ``<base_name>-Instruct``.
        wiki_dir: The registry that holds the VERIFIED records.
        out: The JSONL file to write. It is written again after each condition.
        device: The device of the model.
        seeds: The number of seeds of the random prune.
        unit: What a prune zeroes, one of ``UNITS``.
        loss: The old task of the score, one of ``LOSSES``.
    """
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, base_name)
    runner = _load_runner(base_name, device, "float32", None)
    other = _load_runner(f"{base_name}-Instruct", "cpu", "float32", None)
    delta = deltas(runner.model, other.model)
    del other
    batches = sample_windows(runner.tokenizer)
    held = sample_windows(runner.tokenizer, 2 * SAMPLE_TOKENS)[len(batches) :]
    rows_of = lambda bs: [row.unsqueeze(0) for b in bs for row in b]  # noqa: E731
    windows, held_windows = rows_of(batches), rows_of(held)
    base = base_columns(runner, batches, records)
    cache, firsts = probe_activations(runner, records), first_position_activations(runner, records)
    on_base = {r.submission.identifier.slug: cached_readings(runner, r, cache, firsts)
               for r in records}  # fmt: skip
    lines = base_rows(records, on_base, base_name)
    held_lp = base_logprobs(runner.model, held_windows, device)
    variant = {"unit": unit, "loss": loss}

    def condition_rows(condition, k, seed, neurons):
        """The summary row and the pole rows of one prune, with its held-out KL.

        Args:
            condition: ``none``, ``top`` or ``random``.
            k: The number of neurons pruned.
            seed: The random prune's seed, or ``None``.
            neurons: The neurons to prune, as ``(layer, neuron)``.
        """
        fields = {"model": base_name, "condition": condition, "k": k, "seed": seed,
                  "sample_tokens": SAMPLE_TOKENS, **variant}  # fmt: skip
        change = pruned_change(delta, runner.model, neurons, unit)
        rows = run_condition(runner, batches, records, base, on_base, change, fields)
        with changed(runner.model, change):
            rows[0]["held_kl"] = round(mean_kl(runner.model, held_windows, held_lp, device), 6)
        rows[0]["neurons"] = neurons
        return rows

    # The instruct model comes first: the failed loss needs its failed poles.
    lines += condition_rows("none", 0, None, [])
    score_windows, split = scoring_windows(runner.tokenizer, records, lines, windows, loss)
    if split:
        lines.append({"table": "split", "model": base_name, **variant, **split})
    score_lp = base_logprobs(runner.model, score_windows, device)
    with changed(runner.model, delta):
        layer_scores = scores(runner.model, score_windows, score_lp, device, unit)
    lines.append({"table": "scores", "model": base_name, **variant,
                  "scores": {str(k): [round(float(x), 8) for x in v]
                             for k, v in layer_scores.items()}})  # fmt: skip
    for condition, k, seed, neurons in prune_plan(layer_scores, seeds)[1:]:
        rows = condition_rows(condition, k, seed, neurons)
        lines += rows
        click.echo(json.dumps({k: v for k, v in rows[0].items()
                               if k not in ("flipped", "neurons")}), err=True)  # fmt: skip
        # Every row names its variant, the base rows too, so that the variants can share a file.
        out.write_text("".join(json.dumps(variant | r) + "\n" for r in lines))


if __name__ == "__main__":
    main()
