"""Strict Pydantic v2 schemas for audits, hypotheses and evidence.

Every model is strict (no implicit type coercion) and forbids unknown fields, so a
record that validates here is safe to compile into the public registry. Records on
disk are JSON; load them with ``Model.model_validate_json`` so that ISO-8601 strings
are accepted for datetimes under strict mode.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from openinterp.core.agent_names import same_agent
from openinterp.core.attribution import InputAttribution
from openinterp.core.causal import (  # noqa: F401  (re-exported)
    CausalMetrics,
    CausalPairEffect,
    CausalTarget,
    PatchEffect,
)
from openinterp.core.confounds import (  # noqa: F401  (re-exported)
    ConfoundProfile,
    LabelConfound,
    LabelStat,
    TokenStat,
)
from openinterp.core.costs import ExperimentCosts
from openinterp.core.examples import (  # noqa: F401  (re-exported)
    ActivatingExample,
    Histogram,
    OppositePole,
    check_opposite_pole,
)
from openinterp.core.explanation import ExplanationScores
from openinterp.core.identifiers import (  # noqa: F401  (re-exported)
    ComponentIdentifier,
    FeatureIdentifier,
    HeadIdentifier,
    NeuronIdentifier,
    Pole,
)
from openinterp.core.patchscope import PatchscopeReadout
from openinterp.core.profiles import (  # noqa: F401  (re-exported)
    DictionaryProfile,
    DictionarySite,
    HeadProfile,
)
from openinterp.core.steering import SteeringSamples
from openinterp.core.strict import StrictModel, UnitFloat
from openinterp.core.wiring import Wiring, WiringEdge  # noqa: F401  (re-exported)

VerdictStatus = Literal["VERIFIED", "REJECTED", "NEEDS_REVISION"]
ProbeKind = Literal["hard_negative", "positive_holdout", "counterfactual", "lookahead_consistency"]
SelectionMethod = Literal["random", "attribution", "manual"]

PROTOCOL_VERSION = "0.8"
# Fraction of a unit's max activation counted as "active" (docs/PROTOCOL.md §1). The
# default of ReviewPolicy.activation_threshold; read it here, don't retype 0.25.
ACTIVATION_THRESHOLD = 0.25
# An experiment is named by its verdict timestamp: ``%Y%m%dT%H%M%SZ``.
EXPERIMENT_ID_PATTERN = r"^\d{8}T\d{6}Z$"
# ``provenance.corpus`` of a scan without ``--corpus``: the 552-token built-in sample.
BUILTIN_CORPUS = "builtin:sample_corpus"


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
    files: str = Field(
        description="Glob of the Parquet files sampled, within the dataset; several, "
        "comma-separated, each gave an equal share of the documents."
    )
    text_column: str = "text"
    documents: int = Field(ge=0)
    seed: int
    max_chars: int = Field(gt=0)
    docs_per_row_group: int = Field(gt=0)
    layout: str | None = Field(
        default=None,
        description="How rows were joined into documents (e.g. 'wikitext'); None: one per row.",
    )
    first: bool = False  # the first documents in file order, not a seeded sample (no strip)
    ascii: bool = False  # non-ASCII \u-escaped in the JSONL, as in the registry's corpus
    lang_pattern: str | None = Field(
        default=None,
        description="Regular expression whose group, matched in each file glob, gave its "
        "documents' 'lang' field; None: documents carry no language.",
    )
    license: str | None = None
    sha256: str = Field(description="SHA-256 of the corpus file this describes.")

    def __str__(self) -> str:
        at = f"@{self.revision[:8]}" if self.revision else ""
        how = f"{self.documents} docs, seed {self.seed}"
        how = f"first {self.documents} docs" if self.first else how
        return f"{self.dataset}{at} ({self.files}, {how})"


class CorpusSample(StrictModel):
    """The seeded, token-bounded sample of the corpus a scan read (protocol 0.7).

    Documents are visited in ``random.Random(seed)`` order, and each is taken whole if it
    fits in what is left of ``max_tokens`` (the first always is), then kept in corpus
    order. Tokens are counted by the model's tokenizer, without special tokens.
    Re-drawing with the same corpus, tokenizer, seed and budget gives the same
    documents, which ``sha256`` pins.
    """

    seed: int
    max_tokens: int = Field(gt=0, description="Token budget the sample was drawn to.")
    documents: int = Field(gt=0)
    corpus_documents: int = Field(gt=0, description="Documents in the corpus it was drawn from.")
    tokens: int = Field(ge=0, description="Tokenizer tokens in the sampled documents.")
    sha256: str = Field(
        description='SHA-256 of the sampled documents as JSONL ({"text": ...} per line), '
        "the bytes `openinterp corpus` would write for them."
    )

    def __str__(self) -> str:
        return (
            f"{self.documents} of {self.corpus_documents} docs, {self.tokens:,} tokens "
            f"(≤ {self.max_tokens:,}, seed {self.seed})"
        )


class ExperimentProvenance(StrictModel):
    """Everything needed to re-run the scan that produced a submission."""

    model_revision: str | None = Field(default=None, description="Hugging Face commit hash.")
    dtype: str
    corpus: str = Field(min_length=1, description="Human-readable corpus description.")
    corpus_sha256: str | None = None
    n_snippets: int = Field(ge=0)
    tokens_scanned: int = Field(ge=0)
    snippet_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Window length when documents were scanned as overlapping windows; "
        "None when each was scanned whole.",
    )
    snippet_stride: int | None = Field(
        default=None, gt=0, description="Tokens between the starts of consecutive windows."
    )
    selection: SelectionMethod = Field(
        description="How the neuron was chosen; only 'random' gives unbiased registry statistics."
    )
    seed: int | None = None
    harness_version: str
    environment: RunEnvironment | None = None
    corpus_source: CorpusSource | None = None
    corpus_sample: CorpusSample | None = Field(
        default=None,
        description="The sample of the corpus scanned; None when the whole corpus was "
        "(every record before protocol 0.7).",
    )
    costs: ExperimentCosts | None = Field(
        default=None,
        description="Wall time per stage and LLM usage of the run that produced this record "
        "(reported, never gated).",
    )


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


class VocabularyLens(StrictModel):
    """The pole's output direction read through the logit lens and the Jacobian lens.

    The Jacobian lens (Gurnee et al., 2026) first transports the direction to the final
    layer with the context-averaged Jacobian, so it stays reliable in middle layers
    where the logit lens (J = I) does not. ``overlap`` is the shared fraction of the
    two top-k lists; ``direction_cosine`` is cos(w, J·w), so low values mean later
    layers substantially rotate what this neuron writes. Kurtosis of the vocabulary
    z-scores is high when the pole promotes a small, specific token set.

    The ``*_z`` lists hold each listed token's z-score over the whole vocabulary,
    in the same order. They are scale-invariant, so the two lenses share an axis.
    Records before they were added have ``None``.
    """

    logit_promoted: list[str]
    logit_suppressed: list[str]
    jacobian_promoted: list[str]
    jacobian_suppressed: list[str]
    logit_promoted_z: list[float] | None = None
    logit_suppressed_z: list[float] | None = None
    jacobian_promoted_z: list[float] | None = None
    jacobian_suppressed_z: list[float] | None = None
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

    identifier: ComponentIdentifier
    top_examples: list[ActivatingExample]
    activation_histogram: Histogram | None = None
    first_token_floor: bool | None = None  # v0.8, see NeuronAuditSubmission
    provenance: ExperimentProvenance | None = None
    confounds: ConfoundProfile | None = None
    correlated_neurons: list[CorrelatedNeuron] = Field(default_factory=list)
    vocabulary_lens: VocabularyLens | None = None
    input_attribution: InputAttribution | None = None
    head_profile: HeadProfile | None = None
    model_n_heads: int | None = Field(default=None, gt=0)
    dictionary_profile: DictionaryProfile | None = None
    opposite_pole: OppositePole | None = None

    @model_validator(mode="after")
    def _head_profile(self) -> ScanEvidence:
        check_opposite_pole(self)
        return _require_unit_profile(self)


def _require_unit_profile(model):
    """A head's strength is undefined without its mean output, and a dictionary feature's
    without its dictionary's weights; a neuron needs neither."""
    kind = model.identifier.kind
    for field, owner, unit, why in (
        ("head_profile", "attn_head", "an attention head", "its mean output"),
        ("dictionary_profile", "dictionary_feature", "a dictionary feature", "its dictionary"),
    ):
        present = getattr(model, field) is not None
        units = unit.split(" ", 1)[1] + "s"
        if kind == owner and not present:
            raise ValueError(f"{unit} needs a {field}: {why} defines strength")
        if kind != owner and present:
            raise ValueError(f"{field} is only for {units}")
    return model


class NeuronAuditSubmission(StrictModel):
    """A hypothesis about one unit, frozen once submitted: a neuron pole, an attention head
    (since protocol 0.6) or a dictionary feature. The name predates heads and is kept for
    compatibility."""

    identifier: ComponentIdentifier
    natural_language_explanation: str = Field(min_length=1)
    activating_tokens: list[str]
    counterfactuals: list[CounterfactualPair] = Field(min_length=1)
    lookahead: LookaheadProfile
    submitting_agent_model: str = Field(min_length=1)
    # Evidence gathered by the harness. Optional so hand-written submissions validate.
    max_activation: float | None = Field(default=None, gt=0.0)
    top_examples: list[ActivatingExample] = Field(default_factory=list)
    activation_histogram: Histogram | None = None
    first_token_floor: bool | None = Field(
        default=None,
        description="Protocol 0.8: the unit is active at the first position on at least half "
        "the vocabulary, so it is not measured at a text's first token. None: not judged (a "
        "record from before 0.8), and a review judges it.",
    )
    model_num_layers: int | None = Field(default=None, gt=0)
    model_mlp_width: int | None = Field(default=None, gt=0, description="Neurons per MLP layer.")
    provenance: ExperimentProvenance | None = None
    correlated_neurons: list[CorrelatedNeuron] = Field(default_factory=list)
    confounds: ConfoundProfile | None = None
    vocabulary_lens: VocabularyLens | None = None
    input_attribution: InputAttribution | None = None
    head_profile: HeadProfile | None = None
    model_n_heads: int | None = Field(default=None, gt=0, description="Heads per layer.")
    dictionary_profile: DictionaryProfile | None = None
    patchscope: PatchscopeReadout | None = Field(
        default=None,
        description="The model's own description of the unit's output direction, read after "
        "the proposer answered and shown to neither agent (reported, not gated).",
    )
    steering: SteeringSamples | None = Field(
        default=None,
        description="Neutral prompts continued with the unit clamped high, at baseline and "
        "ablated, taken after the proposer answered and shown to neither agent (reported, "
        "not gated).",
    )
    supersedes: str | None = Field(
        default=None,
        pattern=EXPERIMENT_ID_PATTERN,
        description="Experiment id of the earlier experiment on the same unit that this "
        "submission revises (reported, not gated).",
    )
    opposite_pole: OppositePole | None = Field(
        default=None,
        description="A neuron's other pole, from the same scan: its top examples and "
        "histogram, shown beside this pole on the neuron page (reported, not gated).",
    )
    wiring: Wiring | None = Field(
        default=None,
        description="A neuron pole's top connections by virtual weight, read after the "
        "proposer answered and shown to neither agent (reported, not gated).",
    )

    @model_validator(mode="after")
    def _head_profile(self) -> NeuronAuditSubmission:
        check_opposite_pole(self)
        return _require_unit_profile(self)


class AdversarialTestResult(StrictModel):
    kind: ProbeKind
    prompt: str
    passed: bool
    expected_active: bool | None = None
    activation: float | None = None
    expected_delta: float | None = None
    observed_delta: float | None = None
    detail: str = ""
    # v0.8: False for a one-token prompt to a unit with a first-token floor, which is not
    # measured at the first token. It is recorded but counts in no metric.
    measured: bool = True


class ReviewPolicy(StrictModel):
    """The decision rule a verdict was computed under. See docs/PROTOCOL.md."""

    protocol_version: str = PROTOCOL_VERSION
    activation_threshold: UnitFloat = ACTIVATION_THRESHOLD
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
    # v0.8: a unit active on at least this share of its scan's tokens is default-on, and
    # is not audited as a detector: no hard negative of a few words can stay inactive. One
    # active at the first position on this share of the vocabulary has a first-token floor.
    max_active_share: UnitFloat = 0.5


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
    # v0.8: share of the scan's tokens at or above the "active" line (from its histogram),
    # and whether the unit was measured without each text's first token.
    active_share: UnitFloat | None = None
    first_token_floor: bool | None = None


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
    wiring: Wiring | None = Field(
        default=None,
        description="A neuron pole's top connections by attribution on the review's prompts "
        "where it fired (reported, not gated).",
    )
    notes: str = ""


class AuditRecord(StrictModel):
    """The unit stored in the registry: a submission plus its review verdict."""

    submission: NeuronAuditSubmission
    verdict: VerificationVerdict | None = None

    @property
    def is_verified(self) -> bool:
        return self.verdict is not None and self.verdict.status == "VERIFIED"

    @property
    def scanned_on_builtin_sample(self) -> bool:
        """Scanned on the 552-token built-in sample, too small for a meaningful max
        activation: follow-ups rescan it instead of reusing the scan."""
        provenance = self.submission.provenance
        return provenance is not None and provenance.corpus == BUILTIN_CORPUS

    @property
    def self_reviewed(self) -> bool:
        """The proposer's model also reviewed it: recorded and badged, not gated (§2). Also
        when the record calls it independent but spelled one model two ways (``[1m]``)."""
        if self.verdict is None:
            return False
        if self.verdict.metrics is not None and not self.verdict.metrics.independent_review:
            return True
        return same_agent(self.verdict.reviewer_agent_model, self.submission.submitting_agent_model)
