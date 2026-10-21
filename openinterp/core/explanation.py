"""Schemas for scoring an explanation on held-out text (detection, fuzzing, embedding).

These follow Delphi (Paulo et al., 2025): a judge that sees only the explanation picks
out which held-out windows activate the pole (*detection*), or whether the tokens
marked in a window are the ones that activate it (*fuzzing*); an embedding model ranks
windows by similarity to the explanation (*embedding*). The scores are reported, not
gated (docs/PROTOCOL.md §4).
"""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel, UnitFloat


class ScoredWindow(StrictModel):
    """A held-out window as the judge saw it, measured on its own (like a probe)."""

    text: str
    tokens: list[str]
    activations: list[float]
    quantile: int | None = Field(
        default=None,
        ge=0,
        description="Activation bin of the window's peak among active held-out windows "
        "(0 = weakest); None when the window is inactive.",
    )
    highlighted: list[int] = Field(
        default_factory=list, description="Fuzzing: indices of the tokens marked for the judge."
    )
    label: bool = Field(
        description="What a perfect judge answers: the window is active (detection), or its "
        "marked tokens are exactly its active tokens (fuzzing)."
    )
    judged: bool | None = Field(default=None, description="The judge's answer.")
    similarity: float | None = Field(
        default=None, description="Embedding: cosine between the explanation and the window."
    )


class JudgeScore(StrictModel):
    """How well the judge's answers match the labels.

    ``recall_by_quantile[q]`` is the share of positive windows in activation bin ``q``
    the judge got right (``None`` when the bin had none), so weak activations are not
    hidden behind the top ones.
    """

    balanced_accuracy: UnitFloat
    true_positive_rate: UnitFloat | None
    false_positive_rate: UnitFloat | None
    recall_by_quantile: list[UnitFloat | None]


class ExplanationScores(StrictModel):
    """Delphi-style scores of ``explanation`` on windows the proposer never saw.

    Windows are cut from corpus snippets outside the scan's top examples and measured on
    their own, exactly as the judge reads them. Active windows (peak ≥ ``threshold``) are
    drawn evenly from ``quantiles`` bins of their peak strength, split at
    ``quantile_edges``, and matched with as many inactive ones.
    """

    slug: str = Field(description="The neuron pole scored, as NeuronIdentifier.slug.")
    explanation: str = Field(min_length=1)
    threshold: float = Field(gt=0.0, description="Pole strength counted as active.")
    window_tokens: int = Field(gt=0)
    quantiles: int = Field(gt=0)
    quantile_edges: list[float]
    seed: int
    corpus: str
    corpus_sha256: str | None = None
    candidate_windows: int = Field(ge=0, description="Held-out windows measured.")
    active_windows: int = Field(ge=0, description="Of those, windows reaching the threshold.")
    judge_model: str | None = None
    embedding_model: str | None = None
    detection: JudgeScore | None = None
    fuzzing: JudgeScore | None = None
    embedding_auroc: UnitFloat | None = Field(
        default=None,
        description="AUROC of the embedding similarity for active vs inactive detection windows.",
    )
    detection_windows: list[ScoredWindow] = Field(default_factory=list)
    fuzzing_windows: list[ScoredWindow] = Field(default_factory=list)
