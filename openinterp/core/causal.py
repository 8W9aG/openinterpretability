"""Schemas for causal metrics: how much patching a unit moves the model's prediction.

Recorded in every verdict, never gated (docs/PROTOCOL.md §4). The harness that
measures them is ``openinterp.harness.causal``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openinterp.core.strict import StrictModel, UnitFloat

CausalTarget = Literal["claimed_tokens", "next_token_kl"]


class PatchEffect(StrictModel):
    """One patch of one minimal pair: the unit's effect and its matched null's."""

    effect: float = Field(description="The unit's signed effect (logits or KL, per target).")
    percentile: UnitFloat | None = Field(
        default=None, description="Fraction of the matched units' |effect| below the unit's."
    )
    null_median: float = Field(ge=0.0, description="Median |effect| of the matched units.")
    null_max: float = Field(ge=0.0, description="Largest |effect| of the matched units.")


class CausalPairEffect(StrictModel):
    """Both patches of one non-control minimal pair, for the per-pair causal chart."""

    pair_index: int = Field(ge=0, description="Index into the submission's counterfactuals.")
    sufficiency: PatchEffect
    necessity: PatchEffect


class CausalMetrics(StrictModel):
    """Causal effect of the neuron pole on the model's next-token prediction.

    Since protocol 0.5 each minimal pair is patched both ways and scored on a target that
    favours no particular neuron: the mean logit of a lookahead claim's upcoming tokens
    (``claimed_tokens``), otherwise the next-token KL caused by the patch
    (``next_token_kl``). The pole's effect is ranked against the ``null_neurons``
    same-layer neurons whose strength changes most similarly between the pair's prompts,
    patched the same way and scored on the same target. ``sufficiency_score`` (clean →
    corrupted) and ``necessity_score`` (corrupted → clean) are the median percentile of
    the pole's |effect| among theirs, so 0.5 means "no more influential than a neuron
    that changes as much". Control pairs, where the pole barely moves, are skipped.

    Records with ``target = None`` predate 0.5: their scores are the fraction of the
    clean/corrupted gap in the top-1 next-token logit restored by the patch.
    """

    sufficiency_score: UnitFloat
    necessity_score: UnitFloat
    hydra_mitigated: bool = Field(
        description="True if downstream MLP outputs were frozen to block self-repair (hydra)."
    )
    target: CausalTarget | None = None
    target_tokens: list[str] = Field(
        default_factory=list,
        description="The claimed upcoming tokens scored (``claimed_tokens`` only).",
    )
    sufficiency_effect: float | None = Field(
        default=None,
        description="Median effect of the clean → corrupted patch (logits or KL, per target).",
    )
    necessity_effect: float | None = Field(
        default=None,
        description="Median effect of the corrupted → clean patch (logits or KL, per target).",
    )
    null_neurons: int | None = Field(default=None, ge=0)
    pairs_scored: int | None = Field(
        default=None, ge=0, description="Minimal pairs where the pole moved (controls skipped)."
    )
    pairs: list[CausalPairEffect] = Field(
        default_factory=list,
        description="Per-pair effects behind the medians (empty in older records).",
    )
