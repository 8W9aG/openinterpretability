"""Schemas for scoring an explanation on held-out text (detection, fuzzing, embedding).

These follow Delphi (Paulo et al., 2025). A judge sees only the explanation.

- *detection*: the judge picks out the held-out windows that activate the pole.
- *fuzzing*: the judge says whether the marked tokens in a window activate the pole.
- *embedding*: an embedding model ranks windows by similarity to the explanation.

The scores are reported, not gated (docs/PROTOCOL.md §4).
"""

from __future__ import annotations

from pydantic import Field

from openinterp.core.costs import ExperimentCosts
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
    that the judge got right. It is ``None`` when the bin had no positive windows. It
    shows the weak activations, which the top activations can otherwise hide.
    """

    balanced_accuracy: UnitFloat
    true_positive_rate: UnitFloat | None
    false_positive_rate: UnitFloat | None
    recall_by_quantile: list[UnitFloat | None]


class ControlScore(StrictModel):
    """The judge's scores for a control explanation, on the same windows as the real one.

    A control shows how much of a score comes from the windows and not from the
    explanation. ``empty`` gives the judge no explanation. ``other_unit`` gives it the
    explanation of another unit of the same model. A real explanation should score well
    above both.
    """

    kind: str = Field(description="'empty' or 'other_unit'.")
    explanation: str = Field(description="What the judge saw; '' for the empty control.")
    source_slug: str | None = Field(
        default=None, description="other_unit: the unit whose explanation this is."
    )
    detection_judged: list[bool] = Field(default_factory=list)
    fuzzing_judged: list[bool] = Field(default_factory=list)
    detection: JudgeScore | None = None
    fuzzing: JudgeScore | None = None


class ExplanationScores(StrictModel):
    """Delphi-style scores of ``explanation`` on windows the proposer never saw.

    Windows are cut from corpus snippets outside the scan's top examples and measured on
    their own, exactly as the judge reads them. Active windows (peak ≥ ``threshold``) are
    drawn evenly from ``quantiles`` bins of their peak strength, split at
    ``quantile_edges``, and matched with as many inactive ones.
    """

    slug: str = Field(description="The unit scored: its identifier's slug.")
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
    fuzzing_decoys: str | None = Field(
        default=None,
        description="How fuzzing decoys were marked: 'matched-shape' (the real marks' count "
        "and spacing, moved) since issue #3505. None for older scores, whose decoys were "
        "random tokens that explanation-free rules could tell apart.",
    )
    rule_baseline: dict[str, float] = Field(
        default_factory=dict,
        description="Balanced accuracy of explanation-free rules on these windows, best cut "
        "chosen on them, so an upper bound: detection_length, fuzzing_contiguous, "
        "fuzzing_count. A score means something only above these.",
    )
    controls: list[ControlScore] = Field(
        default_factory=list,
        description="The same windows judged with control explanations (openinterp score "
        "--controls). Each control's answers are scored against these windows' labels.",
    )
    costs: ExperimentCosts | None = Field(
        default=None,
        description="What scoring cost; a review moves it into the record's provenance.",
    )
