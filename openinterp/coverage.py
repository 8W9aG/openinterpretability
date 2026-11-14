"""Registry coverage: how much of each MLP layer's effect on the loss the VERIFIED poles keep.

For one layer, mean-ablating every neuron raises the model's next-token loss. The rise
is what that layer contributes. The coverage of the layer is the share of that rise that
comes back when only the VERIFIED poles keep their real activations:

    recovered = (loss_ablated - loss_kept) / (loss_ablated - loss_clean)

A neuron's activation has two poles: its positive part and its negative part. Each part is
replaced by its own mean over a separate sample of the corpus, unless that pole is
VERIFIED. So a neuron whose positive pole is VERIFIED keeps its positive part only.

The same number of poles, drawn at random from the layer's other poles, gives a baseline.
It is the mean over several draws.
VERIFIED poles that recover no more than random ones explain no more of the loss than
any poles would.

This is a running measure of how much of the model the registry explains. It is
reported, not gated. ``open-interpretability coverage`` writes one row per layer.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import torch

from openinterp.core.paths import DEFAULT_WIKI_DIR
from openinterp.harness.hooks import NeuronHookRunner

CE_CHUNK = 512  # token positions per cross-entropy chunk: Qwen2.5's vocabulary is 151,936


def verified_poles(model_name: str, wiki_dir: Path = DEFAULT_WIKI_DIR) -> dict[int, set]:
    """The neuron poles that stand VERIFIED, as ``{layer: {(neuron_idx, pole)}}``.

    Standing uses the site's rules, errata included (``builder/standing.py``).
    """
    from openinterp.builder.compiler import load_records
    from openinterp.builder.tree import build_tree
    from openinterp.core.errata import load_errata
    from openinterp.core.paths import experiments_dir

    records = load_records(experiments_dir(Path(wiki_dir)))
    out: dict[int, set] = {}
    for model in build_tree(records, load_errata(wiki_dir)):
        if model.name != model_name:
            continue
        for layer in model.layers:
            for neuron in layer.neurons:
                for side, standing in neuron.standings.items():
                    if standing.verified:
                        out.setdefault(layer.layer, set()).add((neuron.idx, side))
    return out


def windows(runner: NeuronHookRunner, texts: Sequence[str], count: int, tokens: int, seed: int):
    """Two disjoint sets of ``count`` windows of ``tokens`` tokens: one to fit, one to test.

    Each window is the start of one document long enough to fill it.
    """
    ids = [runner.tokenizer(t)["input_ids"] for t in texts]
    starts = [row[:tokens] for row in ids if len(row) >= tokens]
    random.Random(seed).shuffle(starts)
    if len(starts) < 2 * count:
        raise ValueError(f"only {len(starts)} documents have {tokens} tokens; need {2 * count}")
    fit, test = starts[:count], starts[count : 2 * count]
    return torch.tensor(fit), torch.tensor(test)


def pole_means(runner: NeuronHookRunner, ids: torch.Tensor, layer: int, batch: int):
    """The mean positive part and mean negative part of each neuron's activation."""
    pos = neg = None
    for chunk in ids.split(batch):
        act = runner.layer_activations(chunk, [layer])[layer].float().flatten(0, -2)
        p, n = act.clamp(min=0).sum(0), act.clamp(max=0).sum(0)
        pos, neg = (p, n) if pos is None else (pos + p, neg + n)
    total = ids.numel()
    return pos / total, neg / total


def keep_poles(means, keep: set, width: int):
    """An intervention that mean-ablates every pole outside ``keep``."""
    m_pos, m_neg = means
    keep_pos = torch.zeros(width, dtype=torch.bool)
    keep_neg = torch.zeros(width, dtype=torch.bool)
    for idx, pole in keep:
        (keep_pos if pole == "positive" else keep_neg)[idx] = True

    def intervene(act: torch.Tensor) -> torch.Tensor:
        dev = act.device
        pos = torch.where(keep_pos.to(dev), act.clamp(min=0), m_pos.to(dev, act.dtype))
        neg = torch.where(keep_neg.to(dev), act.clamp(max=0), m_neg.to(dev, act.dtype))
        return pos + neg

    return intervene


@torch.no_grad()
def mean_loss(runner: NeuronHookRunner, ids: torch.Tensor, batch: int, interventions=None):
    """Mean next-token cross-entropy over ``ids``, with optional MLP interventions."""
    total, count = 0.0, 0
    for chunk in ids.split(batch):
        chunk = chunk.to(runner.device)
        logits = runner.run_with_interventions({"input_ids": chunk}, interventions)
        flat = logits[:, :-1].flatten(0, 1)
        target = chunk[:, 1:].flatten()
        for lo in range(0, target.numel(), CE_CHUNK):
            part = flat[lo : lo + CE_CHUNK].float()
            total += torch.nn.functional.cross_entropy(
                part, target[lo : lo + CE_CHUNK], reduction="sum"
            ).item()
        count += target.numel()
    return total / count


def random_poles(verified: set, width: int, seed: int) -> set:
    """As many poles as ``verified``, drawn from the layer's other poles (or all of them)."""
    others = [
        (i, p) for i in range(width) for p in ("positive", "negative") if (i, p) not in verified
    ]
    return set(random.Random(seed).sample(others, min(len(verified), len(others))))


def recovered(clean: float, ablated: float, kept: float) -> float | None:
    """The share of the ablation's loss rise that keeping the poles brings back."""
    rise = ablated - clean
    return None if rise <= 0 else (ablated - kept) / rise


def layer_row(runner, fit, test, layer: int, verified: set, clean: float, batch: int, seed: int,
              draws: int = 5):  # fmt: skip
    """Measure one layer: ablated, kept-VERIFIED and kept-random losses, and shares.

    The random baseline is the mean over ``draws`` random sets of poles.
    """
    width = runner.mlp_width(layer)
    means = pole_means(runner, fit, layer, batch)
    ablated = mean_loss(runner, test, batch, {layer: keep_poles(means, set(), width)})
    row = {"layer": layer, "width": width, "verified_poles": len(verified),
           "loss_clean": clean, "loss_ablated": ablated}  # fmt: skip
    if verified:
        kept = mean_loss(runner, test, batch, {layer: keep_poles(means, verified, width)})
        shares = []
        for d in range(draws):
            rand = random_poles(verified, width, seed + 1000 * layer + d)
            loss = mean_loss(runner, test, batch, {layer: keep_poles(means, rand, width)})
            shares.append(recovered(clean, ablated, loss))
        valid = [x for x in shares if x is not None]
        row |= {"loss_kept_verified": kept, "recovered": recovered(clean, ablated, kept),
                "recovered_random": sum(valid) / len(valid) if valid else None,
                "recovered_random_draws": shares}  # fmt: skip
    return row
