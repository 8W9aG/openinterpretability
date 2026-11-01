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


class ConceptSource(StrictModel):
    """Where a labelled concept set came from, written by ``openinterp concepts`` next to
    it. Enough to rebuild the same file: the dataset at an exact commit, the split's
    files, the label column and label, the seed and sizes. ``license`` is the dataset
    card's (or, for a Universal Dependencies treebank, the treebank's)."""

    preset: str = Field(examples=["conll2003-ner"])
    dataset: str = Field(min_length=1, examples=["eriktks/conll2003"])
    revision: str = Field(description="Dataset commit hash.")
    files: str = Field(description="Glob of the split's Parquet files, within the dataset.")
    column: str = Field(description="The column labels were read from.", examples=["ner_tags"])
    label: str = Field(
        description="The label marked: tags equal to it, or to it with a BIO prefix or a "
        "subtype (B-PER, nmod:poss), or the documents of that class.",
        examples=["PER"],
    )
    level: Literal["token", "document"] = Field(
        description="token: spans over tagged words; document: a span over the whole text "
        "of each document of the class, half the documents being of it."
    )
    documents: int = Field(ge=0)
    positive_documents: int = Field(ge=0, description="Documents with at least one span.")
    spans: int = Field(ge=0)
    skipped_sentences: int = Field(
        default=0,
        ge=0,
        description="Sentences left out because their words weren't found in their text.",
    )
    seed: int
    max_chars: int = Field(gt=0)
    license: str | None = None
    sha256: str = Field(description="SHA-256 of the concept set file this describes.")

    def __str__(self) -> str:
        return (
            f"{self.dataset}@{self.revision[:8]} ({self.files}, {self.column} = {self.label}: "
            f"{self.spans} spans in {self.positive_documents} of {self.documents} docs)"
        )
