"""Jacobian lens readings of a neuron pole's output direction.

The Jacobian lens (Gurnee et al., 2026, "Verbalizable Representations Form a Global
Workspace in Language Models", arXiv:2607.15495) decodes a layer-ℓ residual vector h as

    lens(h) = softmax(W_U · norm(J_ℓ h)),   J_ℓ = E[∂h_final,t' / ∂h_ℓ,t]

averaged over source positions t, every later position t' ≥ t, and a corpus of
prompts. The logit lens is the special case J_ℓ = I. The two agree in late layers and
diverge earlier, where the logit lens is unreliable.

For one neuron we only need J_ℓ w for its output direction w, not the full matrix.
Adding ε·w to the residual at *every* position changes the final residual at t' by
Σ_{t ≤ t'} J_{t,t'} w to first order (the causal mask stops later positions acting
on earlier ones). Summing that over t' and dividing by the number of (t, t') pairs
gives the paper's average from one perturbed pass per sign (central differences).

The average includes pairs where t' > t contributes nothing, so its overall scale
shrinks with context length (at the last layer it is I · 2/(seq+1)). Every reading
we take (top-k ranking, cosine, kurtosis) is scale-invariant.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from openinterp.core.schemas import Pole, VocabularyLens
from openinterp.harness.hooks import NeuronHookRunner


def _final_residual_module(runner: NeuronHookRunner) -> nn.Module:
    """The module whose input is the final residual stream (pre final norm)."""
    return runner.final_norm() or runner.model.get_output_embeddings()


def _final_residual(
    runner: NeuronHookRunner, module: nn.Module, encoded: dict, layer: int, shift: torch.Tensor
) -> torch.Tensor:
    """Final residual stream [b, seq, d] with ``shift`` added after ``layer``'s MLP."""
    captured: list[torch.Tensor] = []
    handle = module.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().float())
    )
    try:
        runner.run_with_interventions(
            encoded, output_interventions={layer: lambda out: out + shift.to(out.dtype)}
        )
    finally:
        handle.remove()
    return captured[-1]


@torch.no_grad()
def transported_direction(
    runner: NeuronHookRunner,
    prompts: Sequence[str] | torch.Tensor,
    layer: int,
    direction: torch.Tensor,
    *,
    eps: float = 1e-2,
    batch_size: int = 8,
) -> torch.Tensor:
    """Estimate J_ℓ · direction, where ℓ is the residual stream just after ``layer``'s MLP."""
    module = _final_residual_module(runner)
    direction = direction.to(runner.device, torch.float32)
    step = eps / max(float(direction.norm()), 1e-12)
    total = torch.zeros_like(direction, dtype=torch.float64)
    pairs = 0
    batches = (
        [prompts]
        if isinstance(prompts, torch.Tensor)
        else [list(prompts[i : i + batch_size]) for i in range(0, len(prompts), batch_size)]
    )
    for batch in batches:
        encoded = runner.encode(batch)
        mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool()
        plus = _final_residual(runner, module, encoded, layer, step * direction)
        minus = _final_residual(runner, module, encoded, layer, -step * direction)
        delta = (plus - minus) / (2 * step)  # [b, seq, d]
        total += delta[mask].double().sum(0)
        lengths = mask.sum(1)
        pairs += int((lengths * (lengths + 1) // 2).sum())
    return (total / max(pairs, 1)).float()


def _through_final_norm(runner: NeuronHookRunner, vector: torch.Tensor) -> torch.Tensor:
    """Apply the final norm's direction-relevant part: centring for LayerNorm and the
    gain. The bias is a constant offset, unrelated to the direction, so it is dropped."""
    norm = runner.final_norm()
    v = vector.float()
    if isinstance(norm, nn.LayerNorm):
        v = v - v.mean()
    weight = getattr(norm, "weight", None)
    return v * weight.detach().float() if weight is not None else v


def _decode(runner: NeuronHookRunner, vector: torch.Tensor, k: int):
    unembed = runner.model.get_output_embeddings().weight.detach().float()
    scores = unembed @ _through_final_norm(runner, vector).to(unembed.device)
    z = (scores - scores.mean()) / scores.std().clamp(min=1e-12)
    top = scores.topk(k).indices.tolist()
    bottom = (-scores).topk(k).indices.tolist()
    return top, bottom, float((z**4).mean()), _rounded(z[top]), _rounded(z[bottom])


def _rounded(values: torch.Tensor) -> list[float]:
    return [round(float(v), 4) for v in values]


def vocabulary_lens(
    runner: NeuronHookRunner,
    prompts: Sequence[str],
    layer: int,
    neuron_idx: int,
    pole: Pole,
    *,
    k: int = 10,
) -> VocabularyLens:
    """Read the pole's output direction through both the logit lens and the Jacobian lens."""
    direction = runner.neuron_output_direction(layer, neuron_idx).float()
    if pole == "negative":
        direction = -direction
    transported = transported_direction(runner, prompts, layer, direction)

    def words(ids: list[int]) -> list[str]:
        tok = runner.tokenizer
        return [tok.decode([i]) if tok is not None else str(i) for i in ids]

    logit_top, logit_bottom, logit_kurt, logit_top_z, logit_bottom_z = _decode(runner, direction, k)
    jac_top, jac_bottom, jac_kurt, jac_top_z, jac_bottom_z = _decode(runner, transported, k)
    cosine = torch.nn.functional.cosine_similarity(
        direction.to(transported.device), transported, dim=0
    )
    return VocabularyLens(
        logit_promoted=words(logit_top),
        logit_suppressed=words(logit_bottom),
        jacobian_promoted=words(jac_top),
        jacobian_suppressed=words(jac_bottom),
        logit_promoted_z=logit_top_z,
        logit_suppressed_z=logit_bottom_z,
        jacobian_promoted_z=jac_top_z,
        jacobian_suppressed_z=jac_bottom_z,
        overlap=round(len(set(logit_top) & set(jac_top)) / k, 4),
        direction_cosine=round(max(-1.0, min(1.0, float(cosine))), 6),
        logit_kurtosis=round(logit_kurt, 4),
        jacobian_kurtosis=round(jac_kurt, 4),
        prompts=len(prompts),
    )
