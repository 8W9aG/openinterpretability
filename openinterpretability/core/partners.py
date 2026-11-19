"""A partner pole: a same-layer neuron pole that co-varies with the audited pole."""

from __future__ import annotations

from pydantic import Field

from openinterpretability.core.identifiers import Pole
from openinterpretability.core.strict import StrictModel


class CorrelatedNeuron(StrictModel):
    """A same-layer neuron *pole* that co-varies with the audited pole over the scan.

    * ``coactivation_lift``: P(partner active | target active) / P(partner active), with
      add-one smoothing. "Target active" uses the protocol threshold (a fraction of the
      target's max); "partner active" means above the partner pole's own 99th
      percentile. This measures the regime where redundancy matters, when the target fires.
    * ``spearman``: rank correlation of the two pole strengths on a uniform token sample.
      It is robust to heavy tails and massive activations, and ties (zeros) are averaged.
    * ``pearson``: linear correlation with the partner's *signed* activation over every
      token, kept for reference.
    """

    neuron_idx: int = Field(ge=0)
    pole: Pole
    coactivation_lift: float = Field(ge=0.0)
    # 95% Wilson lower bound on P(partner active | target active), over the base rate.
    # Few co-active tokens can inflate the point estimate; partners are ranked by this.
    coactivation_lift_lower: float = Field(ge=0.0)
    coactive_tokens: int = Field(ge=0)
    spearman: float = Field(ge=-1.0, le=1.0)
    pearson: float = Field(ge=-1.0, le=1.0)
    # The partner's strongest distinct tokens in the scan's uniform sample, above its own
    # 99th percentile (#3240). Shown to the proposer for a partner the registry hasn't studied.
    top_tokens: list[str] = Field(default_factory=list, max_length=5)
