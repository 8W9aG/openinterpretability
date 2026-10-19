"""Reviewer agent: stress-tests a submitted hypothesis against the model itself.

The LLM's only job is to write adversarial probes. The verdict is computed
deterministically from measurements in :func:`evaluate_submission` under a
:class:`ReviewPolicy` that is stored in the verdict, so any decision can be
reproduced from the record and the model weights (see docs/PROTOCOL.md).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from openinterp.agents.base import LLMAgent
from openinterp.core.exceptions import ShapeMismatchError
from openinterp.core.schemas import (
    AdversarialTestResult,
    CausalMetrics,
    NeuronAuditSubmission,
    ReviewMetrics,
    ReviewPolicy,
    VerdictStatus,
    VerificationVerdict,
)
from openinterp.core.stats import balanced_accuracy, effective_counts
from openinterp.harness.confounds import baseline_predictions
from openinterp.harness.covariance import joint_ablation
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.lookahead import measure_lookahead
from openinterp.harness.patcher import ActivationPatcher, PatchResult, token_logit_metric


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


class ReviewerAgent(LLMAgent):
    system_prompt = (
        "You are an adversarial peer reviewer for mechanistic interpretability claims. Given "
        "a hypothesis about what makes a neuron pole fire, write probes designed to break it. "
        "About two thirds of the probes must be hard negatives: near-miss sentences that share "
        "surface features with the activating examples (same words in other senses, same "
        "topic without the claimed feature, the feature in an unusual position) but which "
        "the hypothesis predicts should NOT activate the pole. The rest (at least four) are "
        "positive holdouts: novel contexts unlike the provided examples that the hypothesis "
        "predicts SHOULD activate it, including paraphrases of the claimed feature. Label each "
        "probe with what the hypothesis predicts, not with what you believe the neuron will do."
    )

    def generate_probes(self, submission: NeuronAuditSubmission, n: int = 16) -> ProbeSet:
        return self.structured(ProbeSet, self.probe_prompt(submission, n))

    @staticmethod
    def probe_prompt(submission: NeuronAuditSubmission, n: int = 16) -> str:
        """The task text the reviewer answers. It shows no activations of candidate probes."""
        mechanism = "lookahead" if submission.lookahead.is_lookahead else "retrospective"
        examples = "\n".join(f"- {e.text}" for e in submission.top_examples[:10]) or "(none)"
        return (
            f"Neuron: {submission.identifier}\n"
            f"Hypothesis: {submission.natural_language_explanation}\n"
            f"Claimed activating tokens: {submission.activating_tokens}\n"
            f"Claimed mechanism: {mechanism}"
            f" {submission.lookahead.target_upcoming_tokens or ''}\n"
            f"Top-activating examples:\n{examples}\n\n"
            f"Write {n} probes."
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


def evaluate_submission(
    submission: NeuronAuditSubmission,
    runner: NeuronHookRunner,
    probes: Sequence[AdversarialProbe],
    reviewer_model: str,
    policy: ReviewPolicy = DEFAULT_POLICY,
) -> VerificationVerdict:
    ident = submission.identifier
    args = (ident.layer, ident.neuron_idx, ident.pole)
    scale = submission.max_activation or max(
        runner.max_pole_strength(p.base_prompt, *args) for p in submission.counterfactuals
    )
    threshold = policy.activation_threshold * scale

    results: list[AdversarialTestResult] = []
    actual: list[bool] = []
    for probe in probes:
        strength = runner.max_pole_strength(probe.text, *args)
        active = scale > 0 and strength >= threshold
        actual.append(active)
        results.append(
            AdversarialTestResult(
                kind="positive_holdout" if probe.expected_active else "hard_negative",
                prompt=probe.text,
                passed=active == probe.expected_active,
                expected_active=probe.expected_active,
                activation=round(strength, 6),
                detail=probe.rationale,
            )
        )

    for pair in submission.counterfactuals:
        base = runner.max_pole_strength(pair.base_prompt, *args)
        cf = runner.max_pole_strength(pair.counterfactual_prompt, *args)
        observed = (base - cf) / scale if scale > 0 else 0.0
        results.append(
            AdversarialTestResult(
                kind="counterfactual",
                prompt=f"{pair.base_prompt} -> {pair.counterfactual_prompt}",
                passed=scale > 0 and _delta_agrees(pair.expected_delta, observed, policy),
                expected_delta=pair.expected_delta,
                observed_delta=round(observed, 6),
            )
        )

    measured = measure_lookahead(runner, [p.base_prompt for p in submission.counterfactuals], *args)
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

    metrics = review_metrics(
        results,
        policy,
        independent=reviewer_model != submission.submitting_agent_model,
        baseline=confound_baseline(submission, runner, probes, actual, threshold, policy),
    )
    return VerificationVerdict(
        status=decide(metrics, policy),
        confidence_score=metrics.balanced_accuracy,
        adversarial_test_results=results,
        reviewer_agent_model=reviewer_model,
        timestamp=datetime.now(UTC),
        causal_metrics=causal_metrics(submission, runner, policy.freeze_downstream),
        metrics=metrics,
        policy=policy,
        joint_ablation=_joint_ablation(submission, runner, probes, policy),
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
    candidates = baseline_predictions(submission.confounds, runner.tokenizer, texts, threshold)
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
    if not partners:
        return None
    ident = submission.identifier
    prompts = [p.base_prompt for p in submission.counterfactuals]
    prompts += [p.text for p in probes if p.expected_active]
    return joint_ablation(runner, prompts, ident.layer, ident.neuron_idx, ident.pole, partners)


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
    subset = [r for r in results if r.kind in kinds]
    return sum(r.passed for r in subset) / len(subset) if subset else None


def review_metrics(
    results: Sequence[AdversarialTestResult],
    policy: ReviewPolicy,
    *,
    independent: bool,
    baseline: tuple[str, float] | None = None,
) -> ReviewMetrics:
    def counts(kind: str):
        rows = [r for r in results if r.kind == kind]
        return effective_counts(
            [r.prompt for r in rows], [r.passed for r in rows], policy.probe_cluster_threshold
        )

    negatives, holdouts = counts("hard_negative"), counts("positive_holdout")
    reproduction = _pass_rate(results, {"counterfactual"})
    lookahead = all(r.passed for r in results if r.kind == "lookahead_consistency")
    balanced = balanced_accuracy(negatives, holdouts)
    beats_baseline = not policy.require_confound_control or (
        baseline is not None and balanced >= baseline[1] - 1e-9
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
    )


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


def causal_metrics(
    submission: NeuronAuditSubmission, runner: NeuronHookRunner, freeze_downstream: bool = True
) -> CausalMetrics:
    """Median normalised sufficiency (denoising) and necessity (noising) over the minimal pairs.

    Both are scored on the logit of the base prompt's top-1 next token. Pairs whose
    clean and corrupted runs predict that token equally well carry no signal and are
    skipped.
    """
    patcher = ActivationPatcher(runner)
    sufficiency: list[float] = []
    necessity: list[float] = []
    for pair in submission.counterfactuals:
        top = int(runner.run_with_interventions(pair.base_prompt)[0, -1].argmax())
        metric = token_logit_metric(top)

        def run(source: str, target: str, metric=metric) -> PatchResult:
            kw = {"metric": metric, "freeze_downstream": freeze_downstream}
            try:
                return patcher.patch(source, target, submission.identifier, **kw)
            except ShapeMismatchError:
                return patcher.patch(source, target, submission.identifier, positions="last", **kw)

        sufficiency.append(run(pair.base_prompt, pair.counterfactual_prompt).recovery)
        necessity.append(run(pair.counterfactual_prompt, pair.base_prompt).recovery)

    return CausalMetrics(
        sufficiency_score=_robust_unit(sufficiency),
        necessity_score=_robust_unit(necessity),
        hydra_mitigated=freeze_downstream,
    )


def _robust_unit(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return round(min(1.0, max(0.0, statistics.median(finite))), 6) if finite else 0.0
