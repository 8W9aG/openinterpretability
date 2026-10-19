"""Strict Pydantic v2 schemas for audits, hypotheses and evidence.

Every model is strict (no implicit type coercion) and forbids unknown fields, so a
record that validates here is safe to compile into the public registry. Records on
disk are JSON; load them with ``Model.model_validate_json`` so that ISO-8601 strings
are accepted for datetimes under strict mode.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Pole = Literal["positive", "negative"]
VerdictStatus = Literal["VERIFIED", "REJECTED", "NEEDS_REVISION"]
ProbeKind = Literal["hard_negative", "positive_holdout", "counterfactual", "lookahead_consistency"]
SelectionMethod = Literal["random", "attribution", "manual"]

PROTOCOL_VERSION = "0.2"

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class NeuronIdentifier(StrictModel):
    """Addresses one pole of one MLP neuron in an open-weight model."""

    model_name: str = Field(min_length=1, examples=["Qwen/Qwen2.5-0.5B"])
    layer: int = Field(ge=0)
    neuron_idx: int = Field(ge=0)
    pole: Pole

    @property
    def model_slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "--", self.model_name)

    @property
    def slug(self) -> str:
        return f"{self.model_slug}/L{self.layer}/N{self.neuron_idx}-{self.pole}"

    def __str__(self) -> str:
        sign = "+" if self.pole == "positive" else "-"
        return f"{self.model_name} L{self.layer}/N{self.neuron_idx}{sign}"


class CounterfactualPair(StrictModel):
    """A minimal pair predicted to change the neuron's pole activation.

    Deltas are ``strength(base) - strength(counterfactual)`` normalised by the
    neuron's maximum observed activation, so ``1.0`` means "the base prompt fully
    activates the pole and the counterfactual does not activate it at all".
    """

    base_prompt: str = Field(min_length=1)
    counterfactual_prompt: str = Field(min_length=1)
    expected_delta: float
    observed_delta: float | None = None

    @model_validator(mode="after")
    def _prompts_differ(self) -> CounterfactualPair:
        if self.base_prompt == self.counterfactual_prompt:
            raise ValueError("base_prompt and counterfactual_prompt must differ")
        return self


class LookaheadProfile(StrictModel):
    """Whether the neuron prepares specific upcoming tokens rather than reacting to past ones."""

    is_lookahead: bool
    target_upcoming_tokens: list[str] = Field(default_factory=list)
    kl_divergence: float = Field(ge=0.0)


class CausalMetrics(StrictModel):
    """Normalised causal effect of the neuron pole on the model's next-token prediction."""

    sufficiency_score: UnitFloat
    necessity_score: UnitFloat
    hydra_mitigated: bool = Field(
        description="True if downstream MLP outputs were frozen to block self-repair (hydra)."
    )


class Histogram(StrictModel):
    bin_edges: list[float] = Field(min_length=2)
    counts: list[int]

    @model_validator(mode="after")
    def _shape(self) -> Histogram:
        if len(self.counts) != len(self.bin_edges) - 1:
            raise ValueError("counts must have exactly len(bin_edges) - 1 entries")
        return self


class ActivatingExample(StrictModel):
    """A text snippet with per-token pole strengths (non-negative magnitudes)."""

    text: str
    tokens: list[str]
    activations: list[float]
    max_activation: float = Field(ge=0.0)
    max_token_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _aligned(self) -> ActivatingExample:
        if len(self.tokens) != len(self.activations):
            raise ValueError("tokens and activations must be the same length")
        if self.tokens and self.max_token_index >= len(self.tokens):
            raise ValueError("max_token_index out of range")
        return self


class ExperimentProvenance(StrictModel):
    """Everything needed to re-run the scan that produced a submission."""

    model_revision: str | None = Field(default=None, description="Hugging Face commit hash.")
    dtype: str
    corpus: str = Field(min_length=1, description="Human-readable corpus description.")
    corpus_sha256: str | None = None
    n_snippets: int = Field(ge=0)
    tokens_scanned: int = Field(ge=0)
    snippet_tokens: int | None = Field(default=None, gt=0)
    selection: SelectionMethod = Field(
        description="How the neuron was chosen; only 'random' gives unbiased registry statistics."
    )
    seed: int | None = None
    harness_version: str


class NeuronAuditSubmission(StrictModel):
    identifier: NeuronIdentifier
    natural_language_explanation: str = Field(min_length=1)
    activating_tokens: list[str]
    counterfactuals: list[CounterfactualPair] = Field(min_length=1)
    lookahead: LookaheadProfile
    submitting_agent_model: str = Field(min_length=1)
    # Evidence gathered by the harness. Optional so hand-written submissions validate.
    max_activation: float | None = Field(default=None, gt=0.0)
    top_examples: list[ActivatingExample] = Field(default_factory=list)
    activation_histogram: Histogram | None = None
    model_num_layers: int | None = Field(default=None, gt=0)
    provenance: ExperimentProvenance | None = None


class AdversarialTestResult(StrictModel):
    kind: ProbeKind
    prompt: str
    passed: bool
    expected_active: bool | None = None
    activation: float | None = None
    expected_delta: float | None = None
    observed_delta: float | None = None
    detail: str = ""


class ReviewPolicy(StrictModel):
    """The decision rule a verdict was computed under. See docs/PROTOCOL.md."""

    protocol_version: str = PROTOCOL_VERSION
    activation_threshold: UnitFloat = 0.25  # fraction of max activation counted as "active"
    delta_abs_tolerance: float = Field(default=0.10, ge=0.0)
    delta_rel_tolerance: float = Field(default=0.50, ge=0.0)
    min_hard_negatives: int = Field(default=6, ge=0)
    min_holdouts: int = Field(default=3, ge=0)
    min_specificity: UnitFloat = 0.80
    min_sensitivity: UnitFloat = 0.60
    min_counterfactual_reproduction: UnitFloat = 0.75
    reject_below_balanced_accuracy: UnitFloat = 0.60
    require_independent_review: bool = True
    freeze_downstream: bool = True


class ReviewMetrics(StrictModel):
    specificity: UnitFloat | None  # hard negatives correctly inactive
    sensitivity: UnitFloat | None  # positive holdouts correctly active
    balanced_accuracy: UnitFloat
    counterfactual_reproduction: UnitFloat | None
    lookahead_consistent: bool
    independent_review: bool
    gates: dict[str, bool]


class VerificationVerdict(StrictModel):
    status: VerdictStatus
    confidence_score: UnitFloat
    adversarial_test_results: list[AdversarialTestResult]
    reviewer_agent_model: str = Field(min_length=1)
    timestamp: AwareDatetime
    causal_metrics: CausalMetrics | None = None
    metrics: ReviewMetrics | None = None
    policy: ReviewPolicy | None = None
    notes: str = ""


class AuditRecord(StrictModel):
    """The unit stored in the registry: a submission plus its review verdict."""

    submission: NeuronAuditSubmission
    verdict: VerificationVerdict | None = None

    @property
    def is_verified(self) -> bool:
        return self.verdict is not None and self.verdict.status == "VERIFIED"
