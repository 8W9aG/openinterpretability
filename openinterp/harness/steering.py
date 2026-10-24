"""Steering samples: continue neutral prompts with a unit clamped high or ablated.

Every prompt is continued under each intervention, applied at every position of the
prompt and of the continuation:

* baseline: no intervention;
* ablated: the unit's own ablation (a neuron's pole zeroed, or a dictionary latent's
  write removed), as in the causal and lookahead measurements;
* clamped: a neuron's activation set to the pole's signed top activation
  (``max_activation``) times each multiple, or a latent's activation set to that value.

All prompts and interventions decode as one left-padded batch, greedily, so a sample
is deterministic given the weights. The prompts are fixed and unrelated to any
hypothesis, so a coherent drift at 1–2× says what the unit makes the model write; drift
only at 4× is off-distribution and weak evidence. A head writes a subspace with no
"high" setting, so heads get no samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from openinterp.core.steering import SteeringSample, SteeringSamples
from openinterp.harness.hooks import NeuronHookRunner, isolate_pole
from openinterp.harness.units import FeatureUnit, NeuronUnit, Unit

PROMPTS = (
    "The weather today is",
    "I went to the shop and bought",
    "The best way to learn is",
    "Once upon a time,",
    "The report concluded that",
)
MULTIPLES = (1.0, 2.0, 4.0)
MAX_NEW_TOKENS = 24


def steering_interventions(
    runner: NeuronHookRunner, unit: Unit, ablate: torch.Tensor, clamp: torch.Tensor
) -> dict[str, Mapping[int, Any]] | None:
    """``hooks`` keyword arguments that ablate the unit in the batch rows where ``ablate``
    is true and clamp it to ``clamp`` (a strength) where that is not NaN; ``None`` for a
    head."""
    clamped = ~torch.isnan(clamp)
    if isinstance(unit, NeuronUnit):
        idx, pole = unit.ident.neuron_idx, unit.ident.pole
        signed = clamp if pole == "positive" else -clamp

        def steer(act: torch.Tensor) -> torch.Tensor:
            act = act.clone()
            a = act[..., idx]
            a = torch.where(ablate.to(a.device)[:, None], a - isolate_pole(a, pole), a)
            value = signed.to(a.device, a.dtype)[:, None]
            act[..., idx] = torch.where(clamped.to(a.device)[:, None], value, a)
            return act

        return {"interventions": {unit.ident.layer: steer}}
    if isinstance(unit, FeatureUnit):
        dictionary, f = unit.dictionary(runner), unit.ident.feature_idx
        direction = dictionary.w_dec[f]

        def steer_latent(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            a = dictionary.feature(x, f)
            target = torch.where(ablate.to(a.device)[:, None], torch.zeros_like(a), a)
            value = clamp.to(a.device, a.dtype)[:, None]
            target = torch.where(clamped.to(a.device)[:, None], value, target)
            return out + ((target - a).unsqueeze(-1) * direction).to(out.dtype)

        site = unit.profile.input_site
        return {"dictionary_interventions": {unit.ident.layer: (site, steer_latent)}}
    return None


@torch.no_grad()
def _continue(
    runner: NeuronHookRunner,
    rows: Sequence[list[int]],
    interventions: Mapping[str, Mapping[int, Any]],
    max_new_tokens: int,
) -> list[str]:
    """Greedy continuations of every row of token ids, each cut at end-of-text. A model
    with ``generate`` decodes with its cache; one without (a test double) reruns the
    whole sequence each step."""
    tok = runner.tokenizer
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    pad = (eos if pad is None else pad) or 0
    width = max(map(len, rows))
    ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows], device=runner.device)
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=ids.device)
    with runner.hooks(**interventions):
        if hasattr(runner.model, "generate"):
            seq = runner.model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad,
                eos_token_id=eos,
            )
        else:
            seq = ids
            for _ in range(max_new_tokens):
                nxt = runner.forward({"input_ids": seq, "attention_mask": mask})[:, -1].argmax(-1)
                seq = torch.cat([seq, nxt[:, None]], dim=1)
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    texts = []
    for row in seq[:, width:].tolist():
        if eos in row:
            row = row[: row.index(eos)]
        texts.append(tok.decode(row))
    return texts


def steering_samples(
    runner: NeuronHookRunner,
    unit: Unit,
    max_activation: float,
    *,
    prompts: Sequence[str] = PROMPTS,
    multiples: Sequence[float] = MULTIPLES,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> SteeringSamples:
    """Continue every prompt at baseline, ablated, and clamped at each multiple of
    ``max_activation``."""
    if runner.tokenizer is None:
        raise ValueError("steering samples need a tokenizer")
    # Each prompt's rows: baseline, ablated, then clamped at each multiple.
    strengths = [float("nan"), float("nan"), *(max_activation * m for m in multiples)]
    per_prompt = len(strengths)
    clamp = torch.tensor(strengths * len(prompts))
    ablate = torch.tensor(([False, True] + [False] * len(multiples)) * len(prompts))
    interventions = steering_interventions(runner, unit, ablate, clamp)
    if interventions is None:
        raise ValueError(f"{type(unit).__name__} has no clamped-high setting to steer with")
    encoded = [runner.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    rows = [ids for ids in encoded for _ in range(per_prompt)]
    texts = _continue(runner, rows, interventions, max_new_tokens)
    samples = []
    for i, prompt in enumerate(prompts):
        baseline, ablated, *clamped = texts[i * per_prompt : (i + 1) * per_prompt]
        samples.append(
            SteeringSample(prompt=prompt, baseline=baseline, ablated=ablated, clamped=clamped)
        )
    return SteeringSamples(
        clamp_unit=max_activation,
        multiples=[float(m) for m in multiples],
        max_new_tokens=max_new_tokens,
        samples=samples,
    )
