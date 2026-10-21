"""Causal metrics of a unit: sufficiency and necessity, ranked against comparable units.

Recorded in every verdict, never gated (docs/PROTOCOL.md §4). A neuron pole is ranked
against the same layer's neurons, an attention head against the same layer's heads.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import torch

from openinterp.core.schemas import CausalMetrics, CausalTarget, NeuronAuditSubmission
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
    neuron pole's difference is its strength change; a head's is the change in its
    write to the residual stream. The effect is scored there on a target no unit is
    favoured by:

    * ``claimed_tokens``: a lookahead claim's upcoming tokens, as their mean logit minus
      the vocabulary mean (signed);
    * ``next_token_kl``: otherwise, KL(unpatched ‖ patched) of the next-token distribution.

    The same patch is applied, one unit at a time, to the ``null_neurons`` units of the
    same kind and layer (neurons, or the other heads) whose change between the two
    prompts is most similar. A score is the median over pairs of the unit's |effect|
    percentile among theirs: 0.5 means "no more influential than a unit that changes as
    much".

    Pairs where the unit moves by at most ``min_change`` (controls) carry no causal
    signal and are skipped; with none left the result is ``None``.

    Two alternatives read wrongly on real models. Dividing by the whole clean/corrupted
    gap (protocol < 0.5) reads ≈ 0 for any single neuron, because the gap sums every
    component's contribution. Ranking against *random* neurons reads ≈ 1, because minimal
    pairs are built to change this neuron and most others barely change at all.
    """
    from openinterp.harness.heads import patch_heads

    claimed = claimed_token_weights(submission, runner)
    target: CausalTarget = "claimed_tokens" if claimed else "next_token_kl"
    ident = submission.identifier
    is_head = ident.kind == "attn_head"
    patcher = ActivationPatcher(runner)
    scores: dict[str, list[float]] = {"sufficiency": [], "necessity": []}
    effects: dict[str, list[float]] = {"sufficiency": [], "necessity": []}
    nulls: list[int] = []
    for pair in submission.counterfactuals:
        changes = _head_changes if is_head else _neuron_changes
        positions, position, change = changes(
            runner, ident, pair.base_prompt, pair.counterfactual_prompt
        )
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
        for kind, (source, target_prompt) in runs.items():
            units = [ident.index, *null]
            options = {
                "effect": effect,
                "positions": positions,
                "freeze_downstream": freeze_downstream,
            }
            if is_head:
                own, *others = patch_heads(
                    runner, source, target_prompt, ident.layer, units, **options
                )
            else:
                own, *others = patcher.patch_many(
                    source, target_prompt, ident.layer, units, ident.pole, **options
                )
            effects[kind].append(own)
            scores[kind].append(_percentile(abs(own), [abs(o) for o in others]))

    if not effects["sufficiency"]:
        return None
    return CausalMetrics(
        sufficiency_score=_robust_unit(scores["sufficiency"]),
        necessity_score=_robust_unit(scores["necessity"]),
        hydra_mitigated=freeze_downstream,
        target=target,
        target_tokens=claimed[1] if claimed else [],
        sufficiency_effect=round(statistics.median(effects["sufficiency"]), 9),
        necessity_effect=round(statistics.median(effects["necessity"]), 9),
        null_neurons=min(nulls),
        pairs_scored=len(effects["sufficiency"]),
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
