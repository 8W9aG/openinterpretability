"""Reviewer agent: stress-tests a submitted hypothesis against the model itself.

The LLM's only job is to write adversarial probes. The verdict is computed
deterministically from measurements in :func:`evaluate_submission` under a
:class:`ReviewPolicy` that is stored in the verdict, so any decision can be
reproduced from the record and the model weights (see docs/PROTOCOL.md).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial

from pydantic import BaseModel, Field

from openinterp.agents.base import LLMAgent
from openinterp.agents.proposer import (
    FIRST_TOKEN_UNMEASURED,
    format_input_attribution,
    format_review_examples,
)
from openinterp.core import costs
from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.core.schemas import (
    AdversarialTestResult,
    NeuronAuditSubmission,
    ReviewMetrics,
    ReviewPolicy,
    VerdictStatus,
    VerificationVerdict,
)
from openinterp.core.stats import balanced_accuracy, effective_counts
from openinterp.harness.causal import causal_metrics
from openinterp.harness.confounds import baseline_predictions
from openinterp.harness.covariance import joint_ablation
from openinterp.harness.environment import run_environment
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.lookahead import ablation_profile
from openinterp.harness.scan import has_first_token_floor
from openinterp.harness.units import peak_strength, submission_unit
from openinterp.harness.wiring_activations import activation_wiring


class AdversarialProbe(BaseModel):
    text: str
    expected_active: bool = Field(
        description="What the *hypothesis* predicts. Hard negatives are near-misses predicted "
        "inactive; positive holdouts are novel cases predicted active."
    )
    rationale: str


class ProbeSet(BaseModel):
    probes: list[AdversarialProbe] = Field(min_length=10, max_length=20)


DEFAULT_POLICY = ReviewPolicy()

HEAD_ACTIVITY = (
    "A head counts as active at a token when its write to the residual stream there is far "
    "from its average write: what it attends to and copies at that token is unusual. A "
    "probe is active if any of its tokens after the first is. Probes are measured on their "
    "own, so any earlier context a probe relies on (e.g. a repeated phrase) must be inside "
    "it.\n"
    "The head reads tokens, not words. When a hypothesis talks about repeats or matches, "
    "they are of tokens, and a token usually includes the space before it, so ' My' "
    "(after a space) and 'My' (at the start of the text or after a newline) are different "
    "tokens, while '. My' later in the text repeats '. My' earlier, however short or "
    "common. Label each probe by the repeats its tokens make, not by the words you read.\n"
)


class ReviewerAgent(LLMAgent):
    role = "reviewer"
    system_prompt = (
        "You are an adversarial peer reviewer for mechanistic interpretability claims. Given "
        "a hypothesis about what makes a neuron pole (or an attention head, or a sparse "
        "dictionary latent) fire, write probes "
        "designed to break it. "
        "About two thirds of the probes must be hard negatives: near-miss sentences that share "
        "surface features with the activating examples (same words in other senses, same "
        "topic without the claimed feature, the feature in an unusual position) but which "
        "the hypothesis predicts should NOT activate the pole. The rest (at least four) are "
        "positive holdouts: novel contexts unlike the provided examples that the hypothesis "
        "predicts SHOULD activate it, including paraphrases of the claimed feature. Label each "
        "probe with what the hypothesis predicts, not with what you believe the neuron will do."
    )

    def generate_probes(
        self, submission: NeuronAuditSubmission, n: int = 16, full_snippets: bool = False
    ) -> ProbeSet:
        return self.structured(ProbeSet, self.probe_prompt(submission, n, full_snippets))

    @staticmethod
    def probe_prompt(
        submission: NeuronAuditSubmission, n: int = 16, full_snippets: bool = False
    ) -> str:
        """The task text the reviewer answers. It shows no activations of candidate probes."""
        mechanism = "lookahead" if submission.lookahead.is_lookahead else "retrospective"
        is_head = submission.identifier.kind == "attn_head"
        context = ""
        if submission.input_attribution is not None:
            context = format_input_attribution(submission.input_attribution)
        if context:
            context += (
                "Probes are measured on their own, so a probe contains only the context "
                "you write into it: vary the tokens above, not just the peak token.\n"
            )
        return (
            f"{submission.identifier.noun.capitalize()}: {submission.identifier}\n"
            + (HEAD_ACTIVITY if is_head else "")
            + (FIRST_TOKEN_UNMEASURED if submission.first_token_floor else "")
            + f"Hypothesis: {submission.natural_language_explanation}\n"
            f"Claimed activating tokens: {submission.activating_tokens}\n"
            f"Claimed mechanism: {mechanism}"
            f" {submission.lookahead.target_upcoming_tokens or ''}\n"
            + format_review_examples(submission.top_examples, full_snippets)
            + (f"{context}\n" if context else "")
            + f"Write {n} probes."
        )

    @staticmethod
    def token_check_prompt(
        submission: NeuronAuditSubmission, probes: Sequence[AdversarialProbe], tokenizer
    ) -> str:
        """A second pass for the reviewer: its probes as the model's tokens, so it can fix
        labels that it gave by reading words. It shows no activations: only the tokenizer
        runs. For a head, each probe also lists the token runs that repeat an earlier one."""
        is_head = submission.identifier.kind == "attn_head"
        blocks = []
        for n, probe in enumerate(probes):
            ids = tokenizer.encode(probe.text)
            tokens = [tokenizer.decode([i]) for i in ids]
            label = "active" if probe.expected_active else "inactive"
            lines = [
                f"{n}. expected {label}: {probe.text!r}",
                f"   tokens: {' | '.join(repr(t) for t in tokens)}",
            ]
            if is_head:
                runs = [f"{''.join(tokens[a:b])!r} ({b - a})" for a, b in repeated_runs(ids)]
                lines.append(f"   repeated token runs: {', '.join(runs) or 'none'}")
            blocks.append("\n".join(lines))
        repeats = (
            "Each 'repeated token runs' entry is a longest run of tokens that also occurs "
            "earlier in the probe, with its length in tokens.\n"
            if is_head
            else ""
        )
        return (
            ReviewerAgent.probe_prompt(submission, len(probes))
            + "\n\nYou wrote the probes below. Here they are as the model's tokens. "
            + repeats
            + "Check each label against what the hypothesis predicts for these tokens, not "
            "for the words. Where a label is wrong, rewrite the probe so it tests what you "
            "meant (preferred, to keep the share of hard negatives), or relabel it. Leave "
            "the other probes unchanged and write the whole probe set again.\n\n"
            + "\n\n".join(blocks)
        )

    def review(
        self,
        submission: NeuronAuditSubmission,
        runner: NeuronHookRunner,
        *,
        probes: Sequence[AdversarialProbe] | None = None,
        policy: ReviewPolicy = DEFAULT_POLICY,
    ) -> VerificationVerdict:
        if probes is None:
            probes = self.generate_probes(submission).probes
        return evaluate_submission(submission, runner, probes, self.model, policy)


def repeated_runs(ids: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs ``ids[a:b]`` that also occur ending earlier in ``ids``, as ``(a, b)``.

    ``match[j]`` is the longest run ending at ``j`` that ends at some earlier position
    too; a run is reported where it stops growing."""
    match = [0] * len(ids)
    for j in range(len(ids)):
        for i in range(j):
            k = 0
            while k <= i and ids[i - k] == ids[j - k]:
                k += 1
            match[j] = max(match[j], k)
    return [
        (j - match[j] + 1, j + 1)
        for j in range(len(ids))
        if match[j] and (j + 1 == len(ids) or match[j + 1] != match[j] + 1)
    ]


def evaluate_submission(
    submission: NeuronAuditSubmission,
    runner: NeuronHookRunner,
    probes: Sequence[AdversarialProbe],
    reviewer_model: str,
    policy: ReviewPolicy = DEFAULT_POLICY,
    *,
    wire: bool = True,
) -> VerificationVerdict:
    """Measure ``probes`` and the submission's own claims, and decide. ``wire=False``
    skips the activation wiring (reported, not gated), e.g. when reproducing a record."""
    lap = costs.laps()
    unit = submission_unit(submission)
    scale = submission.max_activation or max(
        peak_strength(runner, unit, p.base_prompt) or 0.0 for p in submission.counterfactuals
    )
    threshold = policy.activation_threshold * scale
    if submission.first_token_floor is None:  # judged at the scan since protocol 0.8
        read = partial(unit.strengths, runner)
        floor = has_first_token_floor(runner, read, scale, policy)
        submission = submission.model_copy(update={"first_token_floor": floor})
        unit = submission_unit(submission)

    results: list[AdversarialTestResult] = []
    measured_probes: list[AdversarialProbe] = []
    actual: list[bool] = []
    # A one-token prompt to a unit with a first-token floor has nothing measured: it is
    # recorded as unmeasured and counts in no metric.
    for probe in probes:
        strength = peak_strength(runner, unit, probe.text)
        active = strength is not None and scale > 0 and strength >= threshold
        if strength is not None:
            measured_probes.append(probe)
            actual.append(active)
        results.append(
            AdversarialTestResult(
                kind="positive_holdout" if probe.expected_active else "hard_negative",
                prompt=probe.text,
                passed=strength is not None and active == probe.expected_active,
                expected_active=probe.expected_active,
                activation=None if strength is None else round(strength, 6),
                detail=probe.rationale,
                measured=strength is not None,
            )
        )

    for pair in submission.counterfactuals:
        base = peak_strength(runner, unit, pair.base_prompt)
        cf = peak_strength(runner, unit, pair.counterfactual_prompt)
        both = base is not None and cf is not None
        observed = (base - cf) / scale if both and scale > 0 else 0.0
        results.append(
            AdversarialTestResult(
                kind="counterfactual",
                prompt=f"{pair.base_prompt} -> {pair.counterfactual_prompt}",
                passed=both and scale > 0 and _delta_agrees(pair.expected_delta, observed, policy),
                expected_delta=pair.expected_delta,
                observed_delta=round(observed, 6) if both else None,
                measured=both,
            )
        )

    prompts = [p.base_prompt for p in submission.counterfactuals]
    measured = ablation_profile(runner, prompts, unit.ablation(runner))
    claimed = submission.lookahead.is_lookahead
    results.append(
        AdversarialTestResult(
            kind="lookahead_consistency",
            prompt="; ".join(p.base_prompt for p in submission.counterfactuals),
            passed=measured.is_lookahead == claimed,
            detail=(
                f"claimed {'lookahead' if claimed else 'retrospective'}; measured "
                f"{'lookahead' if measured.is_lookahead else 'retrospective'} "
                f"(KL={measured.kl_divergence:.4g}, tokens={measured.target_upcoming_tokens[:5]})"
            ),
        )
    )

    lap("probe_measurement")
    metrics = review_metrics(
        results,
        policy,
        independent=reviewer_model != submission.submitting_agent_model,
        baseline=confound_baseline(submission, runner, measured_probes, actual, threshold, policy),
        active_share=scan_active_share(submission, policy),
        first_token_floor=submission.first_token_floor,
    )
    lap("gates")
    causal = causal_metrics(
        submission,
        runner,
        freeze_downstream=policy.freeze_downstream,
        null_neurons=policy.causal_null_neurons,
        min_change=policy.delta_abs_tolerance * scale,
    )
    joint = _joint_ablation(submission, runner, probes, policy)
    lap("causal_metrics")
    wiring = _activation_wiring(submission, runner, probes, threshold) if wire else None
    if wire:
        lap("wiring")
    return VerificationVerdict(
        status=decide(metrics, policy),
        confidence_score=metrics.balanced_accuracy,
        adversarial_test_results=results,
        reviewer_agent_model=reviewer_model,
        timestamp=datetime.now(UTC),
        causal_metrics=causal,
        metrics=metrics,
        policy=policy,
        joint_ablation=joint,
        environment=run_environment(runner),
        wiring=wiring,
        notes=_summary(results, metrics),
    )


def _probe_score(
    probes: Sequence[AdversarialProbe],
    predicted: Sequence[bool],
    actual: Sequence[bool],
    policy: ReviewPolicy,
) -> float:
    """Balanced accuracy of ``predicted`` vs ``actual`` activity, split by the hypothesis's
    labels (hard negatives / holdouts), with near-duplicate probes clustered."""

    def subset(expected: bool):
        rows = [
            (p.text, pr == a)
            for p, pr, a in zip(probes, predicted, actual, strict=True)
            if p.expected_active == expected
        ]
        return effective_counts(
            [t for t, _ in rows], [ok for _, ok in rows], policy.probe_cluster_threshold
        )

    return balanced_accuracy(subset(False), subset(True))


def confound_baseline(
    submission: NeuronAuditSubmission,
    runner: NeuronHookRunner,
    probes: Sequence[AdversarialProbe],
    actual: Sequence[bool],
    threshold: float,
    policy: ReviewPolicy,
) -> tuple[str, float] | None:
    """Best trivial explanation of the probes' activity: (name, balanced accuracy).

    ``None`` when the submission has no confound profile (scan predates protocol 0.3).
    """
    if submission.confounds is None or runner.tokenizer is None or not probes:
        return None
    texts = [p.text for p in probes]
    skip_first = bool(submission.first_token_floor)
    candidates = baseline_predictions(
        submission.confounds, runner.tokenizer, texts, threshold, skip_first
    )
    candidates["never active"] = [False] * len(probes)
    scored = {name: _probe_score(probes, pred, actual, policy) for name, pred in candidates.items()}
    best = max(sorted(scored), key=lambda name: scored[name])
    return best, round(scored[best], 6)


def _joint_ablation(submission, runner, probes, policy):
    # Ranked by co-activation lift (the scan's order); Spearman is the fallback signal.
    partners: list[int] = []
    for c in submission.correlated_neurons:
        lifted = c.coactivation_lift_lower >= policy.min_partner_lift and (
            c.coactive_tokens >= policy.min_partner_coactive
        )
        if (lifted or abs(c.spearman) >= policy.min_partner_correlation) and (
            c.neuron_idx not in partners
        ):
            partners.append(c.neuron_idx)
    partners = partners[: policy.joint_ablation_partners]
    if not partners or submission.identifier.kind != "mlp_neuron_pole":
        return None
    ident = submission.identifier
    prompts = [p.base_prompt for p in submission.counterfactuals]
    prompts += [p.text for p in probes if p.expected_active]
    return joint_ablation(runner, prompts, ident.layer, ident.neuron_idx, ident.pole, partners)


def _activation_wiring(submission, runner, probes, threshold):
    """Wiring by attribution at the unit's peak on the prompts it should fire on (the
    counterfactuals' base prompts and the holdouts), counting those where it did."""
    if threshold <= 0:
        return None
    prompts = [p.base_prompt for p in submission.counterfactuals]
    prompts += [p.text for p in probes if p.expected_active]
    try:
        return activation_wiring(runner, submission_unit(submission), prompts, threshold)
    except ArchitectureNotSupportedError:
        return None


def _delta_agrees(expected: float, observed: float, policy: ReviewPolicy) -> bool:
    """Observed Δ must land within max(abs_tol, rel_tol·|expected|) of the prediction.

    Non-control pairs (|expected| > abs_tol) must also have the predicted sign.
    """
    tolerance = max(policy.delta_abs_tolerance, policy.delta_rel_tolerance * abs(expected))
    if abs(observed - expected) > tolerance:
        return False
    if abs(expected) <= policy.delta_abs_tolerance:
        return True
    return math.copysign(1, observed) == math.copysign(1, expected)


def _pass_rate(results: Sequence[AdversarialTestResult], kinds: set[str]) -> float | None:
    subset = [r for r in results if r.kind in kinds and r.measured]
    return sum(r.passed for r in subset) / len(subset) if subset else None


def review_metrics(
    results: Sequence[AdversarialTestResult],
    policy: ReviewPolicy,
    *,
    independent: bool,
    baseline: tuple[str, float] | None = None,
    active_share: float | None = None,
    first_token_floor: bool | None = None,
) -> ReviewMetrics:
    def counts(kind: str):
        rows = [r for r in results if r.kind == kind and r.measured]
        return effective_counts(
            [r.prompt for r in rows], [r.passed for r in rows], policy.probe_cluster_threshold
        )

    negatives, holdouts = counts("hard_negative"), counts("positive_holdout")
    reproduction = _pass_rate(results, {"counterfactual"})
    lookahead = all(r.passed for r in results if r.kind == "lookahead_consistency")
    balanced = balanced_accuracy(negatives, holdouts)
    # The baseline's accuracy is stored rounded, so compare at the same precision: a tie
    # passes (protocol 0.8; before, 10/11 lost to a rounded-up 10/11).
    beats_baseline = not policy.require_confound_control or (
        baseline is not None and round(balanced, 6) >= baseline[1]
    )

    gates = {
        "enough_hard_negatives": negatives.clusters >= policy.min_hard_negatives,
        "enough_holdouts": holdouts.clusters >= policy.min_holdouts,
        "specificity": (negatives.rate or 0.0) >= policy.min_specificity,
        "sensitivity": (holdouts.rate or 0.0) >= policy.min_sensitivity,
        "counterfactual_reproduction": (reproduction or 0.0)
        >= policy.min_counterfactual_reproduction,
        "lookahead_consistent": lookahead,
        "independent_review": independent or not policy.require_independent_review,
        "beats_confound_baseline": beats_baseline,
        "not_default_on": active_share is None or active_share < policy.max_active_share,
    }
    return ReviewMetrics(
        specificity=None if negatives.rate is None else round(negatives.rate, 6),
        sensitivity=None if holdouts.rate is None else round(holdouts.rate, 6),
        balanced_accuracy=round(balanced, 6),
        counterfactual_reproduction=reproduction,
        lookahead_consistent=lookahead,
        independent_review=independent,
        gates=gates,
        effective_hard_negatives=negatives.clusters,
        effective_holdouts=holdouts.clusters,
        confound_baseline=baseline[0] if baseline else None,
        confound_baseline_accuracy=baseline[1] if baseline else None,
        active_share=None if active_share is None else round(active_share, 6),
        first_token_floor=first_token_floor,
    )


def scan_active_share(submission: NeuronAuditSubmission, policy: ReviewPolicy) -> float | None:
    """The share of the scan's tokens the unit is active on, from the submission's
    histogram; ``None`` without one."""
    histogram, scale = submission.activation_histogram, submission.max_activation
    if histogram is None or not scale:
        return None
    return histogram.share_at_least(policy.activation_threshold * scale)


def decide(metrics: ReviewMetrics, policy: ReviewPolicy) -> VerdictStatus:
    """VERIFIED if every gate passes; REJECTED if adequately probed yet near chance."""
    if all(metrics.gates.values()):
        return "VERIFIED"
    adequately_probed = metrics.gates["enough_hard_negatives"] and metrics.gates["enough_holdouts"]
    if adequately_probed and metrics.balanced_accuracy < policy.reject_below_balanced_accuracy:
        return "REJECTED"
    return "NEEDS_REVISION"


def _summary(results: Sequence[AdversarialTestResult], metrics: ReviewMetrics) -> str:
    counts: dict[str, list[int]] = {}
    for r in results:
        passed, total = counts.setdefault(r.kind, [0, 0])
        counts[r.kind] = [passed + r.passed, total + 1]
    tally = "; ".join(f"{kind}: {p}/{t} passed" for kind, (p, t) in counts.items())
    failed = [gate for gate, ok in metrics.gates.items() if not ok]
    return tally + (f". Failed gates: {', '.join(failed)}." if failed else ". All gates passed.")


def probes_from_verdict(verdict: VerificationVerdict) -> list[AdversarialProbe]:
    """Recover the probe set stored in a verdict, to re-evaluate it under another policy."""
    return [
        AdversarialProbe(text=r.prompt, expected_active=bool(r.expected_active), rationale=r.detail)
        for r in verdict.adversarial_test_results
        if r.kind in {"hard_negative", "positive_holdout"}
    ]
