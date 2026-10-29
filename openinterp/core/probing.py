"""Schemas for comparing a neuron pole with linear probes for its concept.

A pole verified for a concept may be the model's whole representation of it or only
part of one. Probes trained on labelled tokens show which: logistic and mass-mean
probes (Marks & Tegmark, 2023) on the layer's MLP hidden activations and on the
residual stream after the layer, and k-sparse probes over the hidden layer's neurons
(Gurnee et al., 2023). Labels come from a rule over text that never looks at the
model, and accuracy is measured on held-out documents, ideally from another source.
Reported, not gated (docs/PROTOCOL.md §4).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openinterp.core.strict import StrictModel, UnitFloat

ProbeSite = Literal["mlp_hidden", "residual"]


class ProbeScore(StrictModel):
    """How well a score separates the held-out tokens' labels."""

    balanced_accuracy: UnitFloat = Field(
        description="At the threshold that maximises balanced accuracy on the training tokens."
    )
    auroc: UnitFloat


class NeuronScore(ProbeScore):
    """The pole's strength alone, read as a one-feature probe."""

    line: float = Field(gt=0.0, description="The protocol's line: 0.25 × the record's scale.")
    balanced_accuracy_at_line: UnitFloat = Field(
        description="Counting a token as the concept when the pole is active at the line, "
        "as a review reads a probe: nothing is fitted."
    )
    rank: int = Field(
        ge=1,
        description="The neuron's rank among the layer's by standardised class-mean "
        "difference on the training tokens (1 = the most separating).",
    )


class SiteProbes(StrictModel):
    """Dense probes on every coordinate of one site."""

    site: ProbeSite
    width: int = Field(gt=0)
    logistic: ProbeScore
    mass_mean: ProbeScore
    logistic_cosine: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine between the logistic probe's direction and the pole's: its "
        "signed basis vector on the hidden layer, its signed output weights on the residual.",
    )
    mass_mean_cosine: float = Field(ge=-1.0, le=1.0)


class SparsePoint(StrictModel):
    """A logistic probe on the k neurons that best separate the training tokens."""

    k: int = Field(ge=1)
    score: ProbeScore
    includes_neuron: bool = Field(description="The audited neuron is among the k.")


class ProbeComparison(StrictModel):
    """One pole against probes for its concept, on the same held-out tokens."""

    slug: str
    explanation: str
    concept: str = Field(description="What the labels mark, as their rule states it.")
    labels: str = Field(description="The labelled file(s) the tokens came from.")
    split: Literal["source", "document"] = Field(
        description="source: trained on one source, tested on another; document: held-out "
        "documents of the same source."
    )
    train_source: str
    test_source: str
    window_tokens: int = Field(gt=0)
    seed: int
    l2: float = Field(ge=0.0, description="Logistic penalty on standardised features.")
    train_positive: int = Field(ge=0)
    train_negative: int = Field(ge=0)
    test_positive: int = Field(ge=0)
    test_negative: int = Field(ge=0)
    neuron: NeuronScore
    sites: list[SiteProbes]
    sparse: list[SparsePoint]
    neurons_needed: int | None = Field(
        default=None,
        description="Smallest k whose sparse probe comes within 0.02 balanced accuracy of "
        "the better dense hidden-layer probe; None if no k tried does.",
    )
