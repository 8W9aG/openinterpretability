"""Every neuron on both sides: activation, output and activation mass (#5230).

The footprint of a change is every unit whose activation or output moves above the noise
floors, whether the registry studies it or not. So this reads every MLP neuron:

- :func:`activation_stats` streams one corpus sample through both sides, a few layers at a
  time. For each neuron, it keeps the sums that give its activation's Pearson correlation
  between the sides. It also keeps the neuron's positive and negative activation mass on
  the first side, for the coverage.
- :func:`output_columns` reads each neuron's effective output column: what the MLP's down
  projection writes for a unit activation. It runs the projection itself, so a patch's
  weight edits count, as hooks or not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

LAYERS_AT_ONCE = 4  # layers read per forward pass, which bounds the memory
COLUMNS_AT_ONCE = 1024  # unit vectors run through the down projection at once


@dataclass
class Stats:
    """Per-neuron sums over the sample, for each layer, ``[d_mlp]`` each."""

    n: int = 0
    sums: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)

    def add(self, layer: int, a: torch.Tensor, b: torch.Tensor) -> None:
        """Add one batch's activations, ``[tokens, d_mlp]`` on each side."""
        a, b = a.double(), b.double()
        parts = {"a": a.sum(0), "b": b.sum(0), "aa": (a * a).sum(0), "bb": (b * b).sum(0),
                 "ab": (a * b).sum(0), "pos": a.clamp(min=0).sum(0),
                 "neg": (-a).clamp(min=0).sum(0)}  # fmt: skip
        old = self.sums.setdefault(layer, {k: torch.zeros_like(v) for k, v in parts.items()})
        for key, value in parts.items():
            old[key] += value

    def pearson(self, layer: int) -> torch.Tensor:
        """Each neuron's correlation between the sides; NaN where one side is constant."""
        s, n = self.sums[layer], self.n
        cov = s["ab"] - s["a"] * s["b"] / n
        var = (s["aa"] - s["a"] ** 2 / n) * (s["bb"] - s["b"] ** 2 / n)
        return cov / var.clamp(min=0).sqrt()


def _acts(side, batch: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    """Each layer's activations on ``side``, first tokens left out, as ``[tokens, d_mlp]``."""
    with side.entered() as runner:
        acts = runner.layer_activations({"input_ids": batch.to(runner.device)}, layers)
    return {la: a[:, 1:].float().cpu().reshape(-1, a.shape[-1]) for la, a in acts.items()}


def activation_stats(first, second, batches: list[torch.Tensor], n_layers: int) -> Stats:
    """The per-neuron sums of every layer over ``batches``, on two sides."""
    stats = Stats()
    for start in range(0, n_layers, LAYERS_AT_ONCE):
        layers = list(range(start, min(start + LAYERS_AT_ONCE, n_layers)))
        from openinterpretability.interference.report import say

        say(f"activations of layers {layers[0]}–{layers[-1]} of {n_layers}, {second.name}")
        tokens = 0
        for batch in batches:
            a, b = _acts(first, batch, layers), _acts(second, batch, layers)
            for layer in layers:
                stats.add(layer, a[layer], b[layer])
            tokens += a[layers[0]].shape[0]
        stats.n = tokens
    return stats


@torch.no_grad()
def output_columns(side, layer: int) -> torch.Tensor:
    """``[d_model, d_mlp]``: what layer ``layer``'s down projection writes per neuron."""
    from openinterpretability.patches.graph import Graph

    with side.entered() as runner:
        module = Graph(runner.model).block(layer, "W_out")
        width = getattr(module, "in_features", None) or module.weight.shape[0]
        param = next(module.parameters())
        zero = module(torch.zeros(1, 1, width, device=param.device, dtype=param.dtype))[0, 0]
        eye = torch.eye(width, device=param.device, dtype=param.dtype)
        parts = []
        for start in range(0, width, COLUMNS_AT_ONCE):
            rows = eye[start : start + COLUMNS_AT_ONCE]
            parts.append((module(rows[None])[0] - zero).float().cpu())
    return torch.cat(parts).T


def column_change(base: torch.Tensor, changed: torch.Tensor) -> torch.Tensor:
    """Each neuron's output change: ``|changed − base| / |base|``, per column."""
    return (changed - base).norm(dim=0) / base.norm(dim=0).clamp(min=1e-12)
