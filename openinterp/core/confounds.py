"""Schemas for confound profiles: how much trivial explanations (token identity,
position, per-token labels such as language) account for a unit's activation.

Recorded in every scan, never gated (docs/PROTOCOL.md §4). The harness that measures
them is ``openinterp.harness.confounds`` and, for labels, ``openinterp.harness.annotations``.
"""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel, UnitFloat


class TokenStat(StrictModel):
    token: str
    token_id: int = Field(ge=0)
    mean_strength: float = Field(ge=0.0)
    count: int = Field(gt=0)


class LabelStat(StrictModel):
    label: str
    count: int = Field(gt=0)
    mean_strength: float = Field(ge=0.0)
    active: int = Field(ge=0, description="Tokens at or above the protocol's activation line.")


class LabelConfound(StrictModel):
    """How much one label set (a token's language, its part of speech) explains of the
    pole's activation: ``r2`` as ``token_r2`` does for token identity, and the binned,
    bias-corrected mutual information between strength and label, in bits. Both are
    over labelled tokens only. ``levels`` lists the most frequent labels."""

    r2: UnitFloat
    mutual_information_bits: float = Field(ge=0.0)
    labelled_tokens: int = Field(ge=0)
    unlabelled_tokens: int = Field(ge=0)
    levels: list[LabelStat]


class ConfoundProfile(StrictModel):
    """How much of the pole's activation simple confounds explain over the scan corpus.

    ``token_r2`` / ``position_r2`` are the variance explained by a per-token-id / per-
    position mean model. ``trigger_tokens`` (highest mean strength, ``count >= min_count``)
    and ``position_means`` drive the trivial baselines a hypothesis must match or beat.
    """

    token_r2: UnitFloat
    position_r2: UnitFloat
    frequency_correlation: float = Field(ge=-1.0, le=1.0)
    min_count: int = Field(gt=0)
    trigger_tokens: list[TokenStat]
    position_means: list[float]
    labels: dict[str, LabelConfound] | None = Field(
        default=None,
        description="Confounds from per-token labels (e.g. 'language'), by label set; "
        "None when the scan had none. Reported, not gated.",
    )
