"""Causal metrics of a unit: sufficiency and necessity, ranked against comparable units.

Recorded in every verdict, never gated (docs/PROTOCOL.md §4). A neuron pole is ranked
against the same layer's neurons, an attention head against the same layer's heads, a
dictionary feature against the same dictionary's other latents.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import torch

from openinterp.core.schemas import (
    CausalMetrics,
    CausalPairEffect,
    CausalTarget,
    NeuronAuditSubmission,
    PatchEffect,
)
from openinterp.harness.hooks import NeuronHookRunner, pole_strength
from openinterp.harness.patcher import (
    ActivationPatcher,
    Positions,
    logit_direction_metric,
    metric_change,
    next_token_kl,
)


def claimed_token_weights(
    submission: NeuronAuditSubmission, runner: NeuronHookRunner
) -> tuple[torch.Tensor, list[str]] | None:
    """Equal vocabulary weights on a lookahead claim's single-token targets, if any resolve."""
    tokenizer = runner.tokenizer
    if not submission.lookahead.is_lookahead or tokenizer is None:
        return None
    vocab = runner.model.get_output_embeddings().weight.shape[0]
    ids: dict[int, str] = {}
    for token in submission.lookahead.target_upcoming_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] < vocab:
            ids.setdefault(encoded[0], token)
    if not ids:
        return None
    weights = torch.zeros(vocab)
    weights[list(ids)] = 1.0 / len(ids)
    return weights, list(ids.values())


def _neuron_changes(runner, ident, base: str, cf: str) -> tuple[Positions, int, torch.Tensor]:
    """Where to patch and score, and every neuron's strength change up to there."""
    b, c = (
        pole_strength(runner.run_with_cache(text, [ident.layer])[1][ident.layer][0], ident.pole)
        for text in (base, cf)
    )  # [seq, d_mlp]
    if b.shape != c.shape:
        return "last", -1, (b[-1] - c[-1]).abs()
    position = int((b[:, ident.neuron_idx] - c[:, ident.neuron_idx]).abs().argmax())
    return "all", position, (b[: position + 1] - c[: position + 1]).abs().amax(0)


def _head_changes(runner, ident, base: str, cf: str) -> tuple[Positions, int, torch.Tensor]:
    """The same for heads, with each head's change in write ``|W_O,h (z_base − z_cf)|``.

    The write change needs no head's mean output, so every head in the layer compares.
    """
    from openinterp.harness.heads import head_write_norms

    z_b, z_c = (runner.head_outputs(text, ident.layer)[0].float() for text in (base, cf))
    if z_b.shape != z_c.shape:
        return "last", -1, head_write_norms(runner, ident.layer, z_b[-1] - z_c[-1])
    changes = head_write_norms(runner, ident.layer, z_b - z_c)  # [seq, n_heads]
    position = int(changes[:, ident.head_idx].argmax())
    return "all", position, changes[: position + 1].amax(0)


def causal_metrics(
    submission: NeuronAuditSubmission,
    runner: NeuronHookRunner,
    *,
    freeze_downstream: bool = True,
    null_neurons: int = 32,
    min_change: float = 0.0,
) -> CausalMetrics | None:
    """Sufficiency and necessity of the unit, each ranked against comparable units.

    Every minimal pair is patched clean → corrupted (sufficiency) and corrupted → clean
    (necessity) at the position where the unit differs most between the two prompts
    (the final position when their lengths differ, since only it can be patched). A
    neuron pole's difference is its strength change; a head's or a dictionary latent's is
    the change in its write to the residual stream. The effect is scored there on a
    target no unit is favoured by:

    * ``claimed_tokens``: a lookahead claim's upcoming tokens, as their mean logit minus
      the vocabulary mean (signed);
    * ``next_token_kl``: otherwise, KL(unpatched ‖ patched) of the next-token distribution.

    The same patch is applied, one unit at a time, to the ``null_neurons`` units of the
    same kind and layer (neurons, the other heads, or the dictionary's other latents) whose
    change between the two prompts is most similar. A score is the median over pairs of
    the unit's |effect| percentile among theirs: 0.5 means "no more influential than a
    unit that changes as much".

    Pairs where the unit moves by at most ``min_change`` (controls) carry no causal
    signal and are skipped; with none left the result is ``None``.

    Two alternatives read wrongly on real models. Dividing by the whole clean/corrupted
    gap (protocol < 0.5) reads ≈ 0 for any single neuron, because the gap sums every
    component's contribution. Ranking against *random* neurons reads ≈ 1, because minimal
    pairs are built to change this neuron and most others barely change at all.
    """
    claimed = claimed_token_weights(submission, runner)
    target: CausalTarget = "claimed_tokens" if claimed else "next_token_kl"
    ident = submission.identifier
    changes, patch = _unit_patching(submission, runner)
    scores: dict[str, list[float]] = {"sufficiency": [], "necessity": []}
    effects: dict[str, list[float]] = {"sufficiency": [], "necessity": []}
    nulls: list[int] = []
    pairs: list[CausalPairEffect] = []
    for index, pair in enumerate(submission.counterfactuals):
        positions, position, change = changes(pair.base_prompt, pair.counterfactual_prompt)
        if float(change[ident.index]) <= min_change:
            continue
        null = _matched_units(change, ident.index, null_neurons)
        nulls.append(len(null))
        effect = (
            metric_change(logit_direction_metric(claimed[0], position))
            if claimed
            else next_token_kl(position)
        )
        runs = {
            "sufficiency": (pair.base_prompt, pair.counterfactual_prompt),
            "necessity": (pair.counterfactual_prompt, pair.base_prompt),
        }
        options = {"effect": effect, "positions": positions, "freeze_downstream": freeze_downstream}
        patched: dict[str, PatchEffect] = {}
        for kind, (source, target_prompt) in runs.items():
            own, *others = patch(source, target_prompt, [ident.index, *null], **options)
            patched[kind], percentile = _patch_effect(own, others)
            effects[kind].append(own)
            scores[kind].append(percentile)
        pairs.append(CausalPairEffect(pair_index=index, **patched))

    if not effects["sufficiency"]:
        return None
    tokens = claimed[1] if claimed else []
    return _metrics(scores, effects, nulls, pairs, target, tokens, freeze_downstream)


def _metrics(scores, effects, nulls, pairs, target, tokens, freeze_downstream) -> CausalMetrics:
    """The medians over the scored pairs."""
    return CausalMetrics(
        sufficiency_score=_robust_unit(scores["sufficiency"]),
        necessity_score=_robust_unit(scores["necessity"]),
        hydra_mitigated=freeze_downstream,
        target=target,
        target_tokens=tokens,
        sufficiency_effect=round(statistics.median(effects["sufficiency"]), 9),
        necessity_effect=round(statistics.median(effects["necessity"]), 9),
        null_neurons=min(nulls),
        pairs_scored=len(effects["sufficiency"]),
        pairs=pairs,
    )


def _patch_effect(own: float, others: list[float]) -> tuple[PatchEffect, float]:
    """The unit's patch effect against its matched units', and its |effect| percentile."""
    null_effects = sorted(abs(o) for o in others)
    percentile = _percentile(abs(own), null_effects)
    effect = PatchEffect(
        effect=round(own, 9),
        percentile=round(percentile, 6) if math.isfinite(percentile) else None,
        null_median=round(statistics.median(null_effects), 9) if null_effects else 0.0,
        null_max=round(null_effects[-1], 9) if null_effects else 0.0,
    )
    return effect, percentile


def _unit_patching(submission: NeuronAuditSubmission, runner: NeuronHookRunner):
    """``changes(base, cf)`` and ``patch(source, target, units, **options)`` for the kind
    of unit ``submission`` is about."""
    from openinterp.harness.features import patch_features, write_changes
    from openinterp.harness.heads import patch_heads
    from openinterp.harness.units import FeatureUnit, submission_unit

    ident, unit = submission.identifier, submission_unit(submission)
    if isinstance(unit, FeatureUnit):
        dictionary, site = unit.dictionary(runner), unit.profile.input_site
        return (
            lambda base, cf: write_changes(runner, dictionary, ident, site, base, cf),
            lambda source, target, units, **options: patch_features(
                runner, dictionary, site, source, target, ident.layer, units, **options
            ),
        )
    if ident.kind == "attn_head":
        return (
            lambda base, cf: _head_changes(runner, ident, base, cf),
            lambda source, target, units, **options: patch_heads(
                runner, source, target, ident.layer, units, **options
            ),
        )
    patcher = ActivationPatcher(runner)
    return (
        lambda base, cf: _neuron_changes(runner, ident, base, cf),
        lambda source, target, units, **options: patcher.patch_many(
            source, target, ident.layer, units, ident.pole, **options
        ),
    )


def _matched_units(change: torch.Tensor, index: int, k: int) -> list[int]:
    """The ``k`` other units whose ``change`` is closest to unit ``index``'s."""
    order = (change - change[index]).abs().argsort(stable=True).tolist()
    return [n for n in order if n != index][:k]


def _percentile(value: float, null: Sequence[float]) -> float:
    """Fraction of ``null`` below ``value``, counting ties as half."""
    if not null:
        return math.nan
    below = sum(n < value for n in null)
    ties = sum(n == value for n in null)
    return (below + 0.5 * ties) / len(null)


def _robust_unit(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return round(min(1.0, max(0.0, statistics.median(finite))), 6) if finite else 0.0
