"""Each MLP pole's effect on the loss, estimated by attribution patching (issue #3486).

Mean-ablating one pole replaces its part of the activation with that part's mean over the
fit windows. The part is the positive part or the negative part. This is the ablation that
``openinterp/coverage.py`` applies to every pole at once. Measuring it pole by pole costs
one forward pass per pole: 73,728 for GPT-2 small. To first order, the change in mean
loss is the sum over tokens of ``dL/dh × (mean − part)``. One backward pass per batch
gives it for every pole of every layer. :func:`exact_effect` measures single poles, to
check the estimate.

The planner draws some explore units in proportion to this effect (``planner_effects``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from openinterp.coverage import mean_loss, pole_means
from openinterp.harness.attribution import _corrupted_grads
from openinterp.harness.hooks import NeuronHookRunner

POLES = ("positive", "negative")


def _loss_sum(ids: torch.Tensor):
    """A metric for ``_corrupted_grads``: the summed next-token cross-entropy on ``ids``."""

    def metric(logits: torch.Tensor) -> torch.Tensor:
        target = ids[:, 1:].to(logits.device).flatten()
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1).float(), target, reduction="sum"
        )

    return metric


def estimate(
    runner: NeuronHookRunner,
    fit: torch.Tensor,
    test: torch.Tensor,
    layers: Sequence[int],
    batch: int,
) -> dict[tuple[int, str], torch.Tensor]:
    """``{(layer, pole): tensor[width]}``: each pole's estimated rise in mean loss on ``test``."""
    means = {layer: pole_means(runner, fit, layer, batch) for layer in layers}
    totals: dict[tuple[int, str], torch.Tensor] = {}
    for chunk in test.split(batch):
        acts, grads = _corrupted_grads(runner, chunk, list(layers), _loss_sum(chunk))
        for layer, act, grad in zip(layers, acts, grads, strict=True):
            act, grad = act.detach().float(), grad.float()
            m_pos, m_neg = (m.to(act.device) for m in means[layer])
            parts = {"positive": (m_pos - act.clamp(min=0)), "negative": (m_neg - act.clamp(max=0))}
            for pole, diff in parts.items():
                value = (diff * grad).sum(dim=(0, 1)).cpu()
                key = (layer, pole)
                totals[key] = value if key not in totals else totals[key] + value
    tokens = test[:, 1:].numel()
    return {key: value / tokens for key, value in totals.items()}


def single_pole(means, idx: int, pole: str):
    """An intervention that mean-ablates only pole ``pole`` of neuron ``idx``."""
    m_pos, m_neg = means

    def intervene(act: torch.Tensor) -> torch.Tensor:
        out = act.clone()
        column = act[..., idx]
        if pole == "positive":
            out[..., idx] = m_pos[idx].to(act.device, act.dtype) + column.clamp(max=0)
        else:
            out[..., idx] = column.clamp(min=0) + m_neg[idx].to(act.device, act.dtype)
        return out

    return intervene


def exact_effect(
    runner: NeuronHookRunner,
    fit: torch.Tensor,
    test: torch.Tensor,
    layer: int,
    idx: int,
    pole: str,
    clean: float,
    batch: int,
) -> float:
    """The measured rise in mean loss on ``test`` when one pole is mean-ablated."""
    means = pole_means(runner, fit, layer, batch)
    return mean_loss(runner, test, batch, {layer: single_pole(means, idx, pole)}) - clean


def top_poles(effects: dict[tuple[int, str], torch.Tensor], k: int) -> list[dict]:
    """The ``k`` poles with the largest estimated effect, largest first."""
    rows = [
        (float(v), layer, idx, pole)
        for (layer, pole), values in effects.items()
        for idx, v in enumerate(values.tolist())
    ]
    rows.sort(reverse=True)
    return [{"layer": la, "neuron": i, "pole": p, "effect": v} for v, la, i, p in rows[:k]]
