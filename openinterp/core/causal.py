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

    Since protocol 0.5, the harness patches each minimal pair in both directions. It
    scores each patch on a target that favours no particular neuron:

    - ``claimed_tokens``: for a lookahead claim, the mean logit of the upcoming tokens.
    - ``next_token_kl``: for other claims, the next-token KL that the patch causes.

    The comparison set is the ``null_neurons`` same-layer neurons whose strength changes
    most like the pole's between the pair's prompts. The harness patches them the same
    way and scores them on the same target. ``sufficiency_score`` (clean → corrupted) and
    ``necessity_score`` (corrupted → clean) give the median percentile of the pole's
    |effect| among theirs. So 0.5 means "no more influential than a neuron that changes
    as much". The harness skips control pairs, where the pole barely moves.

    Records with ``target = None`` predate 0.5. Their scores are the fraction of the
    clean/corrupted gap in the top-1 next-token logit that the patch restores.
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
