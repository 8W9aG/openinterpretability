"""Strict Pydantic v2 schemas for audits, hypotheses and evidence.

Every model is strict (no implicit type coercion) and forbids unknown fields, so a
record that validates here is safe to compile into the public registry. Records on
disk are JSON; load them with ``Model.model_validate_json`` so that ISO-8601 strings
are accepted for datetimes under strict mode.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AwareDatetime, Discriminator, Field, Tag, model_validator

from openinterp.core.explanation import ExplanationScores
from openinterp.core.strict import StrictModel, UnitFloat

Pole = Literal["positive", "negative"]
VerdictStatus = Literal["VERIFIED", "REJECTED", "NEEDS_REVISION"]
ProbeKind = Literal["hard_negative", "positive_holdout", "counterfactual", "lookahead_consistency"]
SelectionMethod = Literal["random", "attribution", "manual"]
CausalTarget = Literal["claimed_tokens", "next_token_kl"]

PROTOCOL_VERSION = "0.5"


class _Component(StrictModel):
    model_name: str = Field(min_length=1, examples=["Qwen/Qwen2.5-0.5B"])
    layer: int = Field(ge=0)

    @property
    def model_slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "--", self.model_name)


class NeuronIdentifier(_Component):
    """Addresses one pole of one MLP neuron in an open-weight model.

    The ``mlp_neuron_pole`` case of :data:`ComponentIdentifier`. ``kind`` defaults, so
    records written before it existed still validate.
    """

    kind: Literal["mlp_neuron_pole"] = "mlp_neuron_pole"
    neuron_idx: int = Field(ge=0)
    pole: Pole

    @property
    def slug(self) -> str:
        return f"{self.model_slug}/L{self.layer}/N{self.neuron_idx}-{self.pole}"

    def __str__(self) -> str:
        sign = "+" if self.pole == "positive" else "-"
        return f"{self.model_name} L{self.layer}/N{self.neuron_idx}{sign}"


class HeadIdentifier(_Component):
    """Addresses one attention head: its slice of the attention output projection's input.

    Heads have no sign, so no pole. They can be identified and hooked
    (:meth:`NeuronHookRunner.head_outputs`), but not yet audited: submissions still take
    a :class:`NeuronIdentifier`.
    """

    kind: Literal["attn_head"] = "attn_head"
    head_idx: int = Field(ge=0)

    @property
    def slug(self) -> str:
        return f"{self.model_slug}/L{self.layer}/H{self.head_idx}"

    def __str__(self) -> str:
        return f"{self.model_name} L{self.layer}/H{self.head_idx}"


def _component_kind(value) -> str:
    kind = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", None)
    return kind or "mlp_neuron_pole"  # records predating `kind` are neuron poles


ComponentIdentifier = Annotated[
    Annotated[NeuronIdentifier, Tag("mlp_neuron_pole")]
    | Annotated[HeadIdentifier, Tag("attn_head")],
    Discriminator(_component_kind),
]
"""Any auditable unit: an MLP neuron pole or an attention head, told apart by ``kind``."""


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


class RunEnvironment(StrictModel):
    """Where a measurement ran. Reproducing a record pins the model revision and dtype;
    the device and library versions explain drift that a different setup can cause."""

    device: str
    dtype: str = Field(description="Dtype of the loaded weights, e.g. float32.")
    model_revision: str | None = Field(default=None, description="Hugging Face commit hash.")
    openinterp_version: str
    torch_version: str
    transformers_version: str | None = None


class CorpusSource(StrictModel):
    """Where a scan corpus came from, written by ``openinterp corpus`` next to the corpus.

    Enough to rebuild the same documents: the dataset at an exact commit, the files and
    column read, the seed and sizes. ``license`` is the dataset card's licence, so pages
    that show excerpts can attribute them.
    """

    dataset: str = Field(min_length=1, examples=["HuggingFaceFW/fineweb-edu"])
    revision: str | None = Field(default=None, description="Dataset commit hash.")
    files: str = Field(description="Glob of the Parquet files sampled, within the dataset.")
    text_column: str = "text"
    documents: int = Field(ge=0)
    seed: int
    max_chars: int = Field(gt=0)
    docs_per_row_group: int = Field(gt=0)
    license: str | None = None
    sha256: str = Field(description="SHA-256 of the corpus file this describes.")

    def __str__(self) -> str:
        at = f"@{self.revision[:8]}" if self.revision else ""
        return f"{self.dataset}{at} ({self.files}, {self.documents} docs, seed {self.seed})"


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
    environment: RunEnvironment | None = None
    corpus_source: CorpusSource | None = None


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


class TokenStat(StrictModel):
    token: str
    token_id: int = Field(ge=0)
    mean_strength: float = Field(ge=0.0)
    count: int = Field(gt=0)


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


class VocabularyLens(StrictModel):
    """The pole's output direction read through the logit lens and the Jacobian lens.

    The Jacobian lens (Gurnee et al., 2026) first transports the direction to the final
    layer with the context-averaged Jacobian, so it stays reliable in middle layers
    where the logit lens (J = I) does not. ``overlap`` is the shared fraction of the
    two top-k lists; ``direction_cosine`` is cos(w, J·w), so low values mean later
    layers substantially rotate what this neuron writes. Kurtosis of the vocabulary
    z-scores is high when the pole promotes a small, specific token set.
    """

    logit_promoted: list[str]
    logit_suppressed: list[str]
    jacobian_promoted: list[str]
    jacobian_suppressed: list[str]
    overlap: UnitFloat
    direction_cosine: float = Field(ge=-1.0, le=1.0)
    logit_kurtosis: float = Field(ge=0.0)
    jacobian_kurtosis: float = Field(ge=0.0)
    prompts: int = Field(ge=0)


class JointAblation(StrictModel):
    """Next-token KL when ablating the pole, its correlated partners, or both.

    ``redundancy_ratio`` = (kl_joint − kl_partners) / kl_single: how much more the pole
    matters once its correlated partners are gone (> 1 suggests redundancy).
    """

    partners: list[int]
    kl_single: float = Field(ge=0.0)
    kl_partners: float = Field(ge=0.0)
    kl_joint: float = Field(ge=0.0)
    redundancy_ratio: float | None = None


class ScanEvidence(StrictModel):
    """What ``openinterp audit --evidence-out`` measured, for a proposer working offline.

    It holds everything a submission needs except the hypothesis, so ``openinterp audit
    --evidence <file> --hypothesis <file>`` can build the submission without rescanning.
    """

    identifier: NeuronIdentifier
    top_examples: list[ActivatingExample]
    activation_histogram: Histogram | None = None
    provenance: ExperimentProvenance | None = None
    confounds: ConfoundProfile | None = None
    correlated_neurons: list[CorrelatedNeuron] = Field(default_factory=list)
    vocabulary_lens: VocabularyLens | None = None


class NeuronAuditSubmission(StrictModel):
    identifier: NeuronIdentifier

    @model_validator(mode="before")
    @classmethod
    def _neurons_only(cls, data):
        ident = data.get("identifier") if isinstance(data, dict) else None
        if _component_kind(ident or {}) == "attn_head":
            raise ValueError(
                "attention heads can be identified and hooked, but not audited yet: "
                "submissions take an MLP neuron pole"
            )
        return data

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
    model_mlp_width: int | None = Field(default=None, gt=0, description="Neurons per MLP layer.")
    provenance: ExperimentProvenance | None = None
    correlated_neurons: list[CorrelatedNeuron] = Field(default_factory=list)
    confounds: ConfoundProfile | None = None
    vocabulary_lens: VocabularyLens | None = None


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
    # v0.4: proposer/reviewer independence is recorded but no longer gated by default.
    require_independent_review: bool = False
    freeze_downstream: bool = True
    # v0.5: causal effects are ranked against this many same-layer neurons whose strength
    # changes as much between each minimal pair.
    causal_null_neurons: int = Field(default=32, ge=1)
    # v0.3: near-duplicate probes (word-set Jaccard >= this) count as one observation.
    probe_cluster_threshold: UnitFloat = 0.5
    # v0.3: the hypothesis must match or beat trivial token/position baselines.
    require_confound_control: bool = True
    joint_ablation_partners: int = Field(default=3, ge=0)
    # Partners for joint ablation: lower-bound co-activation lift >= min_partner_lift on
    # at least min_partner_coactive tokens, or |Spearman| >= min_partner_correlation.
    min_partner_correlation: UnitFloat = 0.2
    min_partner_lift: float = Field(default=2.0, ge=1.0)
    min_partner_coactive: int = Field(default=5, ge=1)


class ReviewMetrics(StrictModel):
    specificity: UnitFloat | None  # hard negatives correctly inactive
    sensitivity: UnitFloat | None  # positive holdouts correctly active
    balanced_accuracy: UnitFloat
    counterfactual_reproduction: UnitFloat | None
    lookahead_consistent: bool
    independent_review: bool
    gates: dict[str, bool]
    effective_hard_negatives: int | None = None  # clusters of near-duplicate probes
    effective_holdouts: int | None = None
    confound_baseline_accuracy: UnitFloat | None = None  # best trivial baseline, same probes
    confound_baseline: str | None = None


class VerificationVerdict(StrictModel):
    status: VerdictStatus
    confidence_score: UnitFloat
    adversarial_test_results: list[AdversarialTestResult]
    reviewer_agent_model: str = Field(min_length=1)
    timestamp: AwareDatetime
    causal_metrics: CausalMetrics | None = None
    metrics: ReviewMetrics | None = None
    policy: ReviewPolicy | None = None
    joint_ablation: JointAblation | None = None
    environment: RunEnvironment | None = None
    explanation_scores: ExplanationScores | None = None
    notes: str = ""


class AuditRecord(StrictModel):
    """The unit stored in the registry: a submission plus its review verdict."""

    submission: NeuronAuditSubmission
    verdict: VerificationVerdict | None = None

    @property
    def is_verified(self) -> bool:
        return self.verdict is not None and self.verdict.status == "VERIFIED"
