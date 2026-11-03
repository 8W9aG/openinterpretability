"""Schemas for input attribution: which context tokens drive a pole at its peak.

Integrated gradients (Sundararajan et al., 2017) run over the input token embeddings.
They attribute the pole's activation at each top example's peak token to every token up
to the peak. The baseline replaces every token with one reference token. A second check
replaces each token alone with the reference token and measures the pole again.
Reported, not gated (docs/PROTOCOL.md §4).
"""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel, UnitFloat


class AttributedToken(StrictModel):
    """One context token's contribution to the pole at the peak.

    Both numbers are fractions of one gap: the pole's activation on the example minus
    its activation on the all-baseline input. So ``attribution`` sums to about 1 over
    the tokens, and you can compare the two numbers directly.
    """

    offset: int = Field(le=0, description="Position relative to the peak token (0 = the peak).")
    token: str
    attribution: float = Field(description="Integrated-gradients share of the gap.")
    ablation_effect: float | None = Field(
        default=None,
        description="Drop in activation when only this token is replaced, as a share of the gap.",
    )


class ContextExample(StrictModel):
    """Input attribution for one top example."""

    example: int = Field(ge=0, description="Index into the scan's top examples.")
    peak_token: str
    tokens: int = Field(gt=0, description="Tokens attributed: the example up to its peak.")
    steps: int = Field(gt=0, description="Integration steps used.")
    converged: bool = Field(
        description="completeness_error within the tolerance; only these count in the shares."
    )
    completeness_error: float = Field(
        ge=0.0, description="|Σ attribution − 1|: how far the path integral is from converged."
    )
    top: list[AttributedToken] = Field(description="Largest |attribution| first.")


class DistanceShare(StrictModel):
    """Share of total |attribution| on tokens ``min_distance``..``max_distance`` back."""

    min_distance: int = Field(ge=0)
    max_distance: int | None = Field(default=None, description="None: no upper bound.")
    share: UnitFloat

    @property
    def label(self) -> str:
        """The distance band as the page shows it: ``0``, ``1–3`` or ``8+``."""
        if self.max_distance is None:
            return f"{self.min_distance}+"
        if self.max_distance == self.min_distance:
            return str(self.min_distance)
        return f"{self.min_distance}–{self.max_distance}"


class InputAttribution(StrictModel):
    """Where in the context a pole's activation comes from, over its top examples.

    ``distances`` splits each example's |attribution| by how far back each token is from
    the peak, then averages the shares over examples. A high share at distance 0 means
    that the pole fires on the current token. A high share far back means that it fires
    because of earlier context.

    ``ablation_agreement`` is Spearman's ρ between attribution and single-token ablation
    effect. It uses every checked token: the peak, the tokens just before it and the most
    attributed ones. A low value means that the linear path integral misreads this pole,
    so trust the ablation effects.

    Both use only the examples whose path integral converged. With none, they are empty.
    """

    method: str = "integrated_gradients"
    baseline_token: str
    tolerance: float = Field(gt=0.0, description="Largest completeness error accepted.")
    examples: list[ContextExample]
    distances: list[DistanceShare]
    mean_distance: float | None = Field(
        default=None, ge=0.0, description="|attribution|-weighted mean tokens back from the peak."
    )
    ablation_agreement: float | None = Field(default=None, ge=-1.0, le=1.0)

    @property
    def converged(self) -> int:
        """How many examples converged; only these count in the shares."""
        return sum(ex.converged for ex in self.examples)

    @property
    def current_token_share(self) -> float | None:
        """The share at distance 0, the first band, or ``None`` with no converged example."""
        return self.distances[0].share if self.distances else None
