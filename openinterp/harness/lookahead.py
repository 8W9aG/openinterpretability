"""Distinguish retrospective neurons from lookahead (next-token preparing) neurons.

Two complementary measurements on native weights:

* :func:`output_token_projection` reads the neuron's output direction through the
  unembedding (a logit-lens approximation that skips the final norm) to list the
  tokens the pole directly promotes.
* :func:`measure_lookahead` zero-ablates the pole on real prompts and compares the
  next-token distributions. A lookahead neuron's ablation moves probability mass
  in a concentrated way onto/off a small set of upcoming tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from openinterp.core.schemas import LookaheadProfile, Pole
from openinterp.harness.hooks import Inputs, NeuronHookRunner, isolate_pole


def output_token_projection(
    runner: NeuronHookRunner, layer: int, neuron_idx: int, pole: Pole, k: int = 10
) -> list[tuple[int, float]]:
    """Top-``k`` ``(token_id, score)`` pairs promoted by the pole via the direct path."""
    unembed = runner.model.get_output_embeddings().weight.detach()  # [vocab, d_model]
    direction = runner.neuron_output_direction(layer, neuron_idx).to(unembed.dtype)
    scores = unembed @ direction
    if pole == "negative":
        scores = -scores
    top = scores.topk(min(k, scores.numel()))
    return list(zip(top.indices.tolist(), top.values.tolist(), strict=True))


@torch.no_grad()
def measure_lookahead(
    runner: NeuronHookRunner,
    prompts: Inputs,
    layer: int,
    neuron_idx: int,
    pole: Pole,
    *,
    k: int = 10,
    kl_threshold: float = 1e-3,
    concentration_threshold: float = 0.5,
) -> LookaheadProfile:
    """Zero-ablate the pole on ``prompts`` and profile its effect on next-token predictions.

    ``kl_divergence`` is KL(clean || ablated) averaged over non-padding positions.
    The neuron is flagged as lookahead when that KL exceeds ``kl_threshold`` and the
    ``k`` tokens losing the most probability account for at least
    ``concentration_threshold`` of the total probability mass moved.
    """

    def ablate(act: torch.Tensor) -> torch.Tensor:
        act = act.clone()
        act[..., neuron_idx] = act[..., neuron_idx] - isolate_pole(act[..., neuron_idx], pole)
        return act

    encoded = runner.encode(prompts)
    mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool()
    clean = F.log_softmax(runner.run_with_interventions(encoded).float(), dim=-1)[mask]
    ablated = F.log_softmax(
        runner.run_with_interventions(encoded, {layer: ablate}).float(), dim=-1
    )[mask]

    kl = F.kl_div(ablated, clean, log_target=True, reduction="batchmean").clamp(min=0.0)
    prob_shift = (clean.exp() - ablated.exp()).mean(dim=0)  # positive = neuron supports token
    total_moved = prob_shift.abs().sum()
    top = prob_shift.topk(min(k, prob_shift.numel()))
    concentration = float(top.values.clamp(min=0).sum() / total_moved) if total_moved > 0 else 0.0

    tokenizer = runner.tokenizer
    tokens = [
        tokenizer.decode([i]) if tokenizer is not None else str(i)
        for i, v in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        if v > 0
    ]
    return LookaheadProfile(
        is_lookahead=float(kl) >= kl_threshold and concentration >= concentration_threshold,
        target_upcoming_tokens=tokens,
        kl_divergence=float(kl),
    )
