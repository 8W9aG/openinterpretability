"""Reviewer agent: stress-tests a submitted hypothesis against the model itself.

The LLM's only job is to write adversarial probes. :func:`evaluate_submission` computes
the verdict deterministically from measurements. It uses a :class:`ReviewPolicy`, which
is stored in the verdict. So any decision can be reproduced from the record and the model
weights (see docs/PROTOCOL.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from openinterpretability import prior_art, wordnet
from openinterpretability.agents.base import LLMAgent
from openinterpretability.agents.probe_hints import (  # noqa: F401  (re-exported)
    LANGUAGE_PROBES,
    POS_CHECK,
    POS_PROBES,
    SYNTAX_PROBES,
    has_language_labels,
    has_pos_labels,
    has_syntax_labels,
    pos_line,
)
from openinterpretability.agents.proposer import (
    FIRST_TOKEN_UNMEASURED,
    format_input_attribution,
    format_review_examples,
)
from openinterpretability.agents.review_measurements import (  # noqa: F401  (_delta_agrees re-exported)
    _activation_wiring,
    _delta_agrees,
    causal_readouts,
    lookahead_result,
    measure_counterfactuals,
    measure_probes,
    settle_first_token_floor,
)
from openinterpretability.core import costs
from openinterpretability.core.agent_names import same_agent
from openinterpretability.core.fence import FENCE_RULE, fence
from openinterpretability.core.rivals import compare, rival_prompt
from openinterpretability.core.schemas import (
    MAX_PROBES,
    AdversarialTestResult,
    NeuronAuditSubmission,
    ReviewMetrics,
    ReviewPolicy,
    VerdictStatus,
    VerificationVerdict,
    scan_active_share,
)
from openinterpretability.core.stats import balanced_accuracy, effective_counts
from openinterpretability.harness.confounds import (
    LENGTH_RATIO_WARN,  # noqa: F401 - re-exported
    baseline_predictions,
    length_predictions,
    length_ratio,  # noqa: F401 - re-exported
    length_warning,
)
from openinterpretability.harness.environment import run_environment
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.inverse_claims import INVERSE_RULE_VERSION, count_line
from openinterpretability.inverse_claims import gates as claim_gates


class AdversarialProbe(BaseModel):  # noqa: D101 (a docstring would join the JSON schema that the agent reads)
    text: str
    expected_active: bool = Field(
        description="What the *hypothesis* predicts. Hard negatives are near-misses predicted "
        "inactive; positive holdouts are novel cases predicted active."
    )
    rationale: str
    rival_expected_active: list[bool] = Field(
        default_factory=list,
        description="What each rival reading predicts for this probe, in the order the prompt "
        "lists them. Empty if the prompt lists none.",
    )


def split(n: int) -> tuple[int, int]:
    """The hard negatives and positive holdouts that a set of ``n`` probes asks for."""
    holdouts = max(4, round(n / 3))  # the system prompt asks for at least four
    return n - holdouts, holdouts


class ProbeSet(BaseModel):  # noqa: D101 (a docstring would join the JSON schema that the agent reads)
    probes: list[AdversarialProbe] = Field(min_length=10, max_length=MAX_PROBES)


DEFAULT_POLICY = ReviewPolicy()

# Probes in two groups of very different lengths can be told apart by length alone. Since
# protocol 0.9 the confound baseline includes that rule (issue #3525), so the reviewer is
# asked to match lengths (and warned: harness.confounds.length_warning).
LENGTH_MATCHED = (
    " Give the hard negatives and the positive holdouts similar lengths in tokens. The "
    "review compares the hypothesis with a rule that reads only a probe's length. If your "
    "holdouts are much longer than your negatives, that rule can explain them as well as "
    "the hypothesis does, and the hypothesis fails."
)
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
    """The reviewer: writes probes designed to break the hypothesis."""

    role = "reviewer"
    system_prompt = (
        "You are an adversarial peer reviewer for mechanistic interpretability claims. Given "
        "a hypothesis about what makes a neuron pole (or an attention head, or a sparse "
        "dictionary latent) fire, write probes "
        "designed to break it. "
        "Two thirds of the probes must be hard negatives: near-miss sentences that share "
        "surface features with the activating examples (same words in other senses, same "
        "topic without the claimed feature, the feature in an unusual position) but which "
        "the hypothesis predicts should NOT activate the pole. The rest (at least four) are "
        "positive holdouts: novel contexts unlike the provided examples that the hypothesis "
        "predicts SHOULD activate it, including paraphrases of the claimed feature. Label each "
        "probe with what the hypothesis predicts, not with what you believe the neuron will do. "
        + FENCE_RULE
    )

    def generate_probes(
        self,
        submission: NeuronAuditSubmission,
        n: int = MAX_PROBES,
        full_snippets: bool = False,
        related: str = "",
    ) -> ProbeSet:
        """Ask the reviewer for ``n`` probes."""
        prompt = self.probe_prompt(submission, n, full_snippets, related)
        return self.structured(ProbeSet, prompt)

    @staticmethod
    def probe_prompt(
        submission: NeuronAuditSubmission,
        n: int = MAX_PROBES,
        full_snippets: bool = False,
        related: str = "",
    ) -> str:
        """Return the task text that the reviewer answers.

        It shows no activations of candidate probes. ``related`` is what other analyses
        link to the unit, so that the reviewer can probe where they disagree. That is
        published latents aligned with the unit
        (:func:`openinterpretability.crossref.format_for_reviewer`), and its studied sister neurons
        (:func:`openinterpretability.sisters.format_sisters`). The submission's prior art is always
        shown (:func:`openinterpretability.prior_art.format_for_reviewer`).
        """
        mechanism = "lookahead" if submission.lookahead.is_lookahead else "retrospective"
        claim = (
            f"Hypothesis: {submission.natural_language_explanation}\n"
            f"Claimed activating tokens: {submission.activating_tokens}\n"
            f"Claimed mechanism: {mechanism}"
            f" {submission.lookahead.target_upcoming_tokens or ''}"
        )
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
            + f"The submitted hypothesis, which your probes test:\n{fence(claim)}\n"
            + rival_prompt(submission.rival_explanations, n)
            + format_review_examples(submission.top_examples, full_snippets)
            + (LANGUAGE_PROBES if has_language_labels(submission) else "")
            + (SYNTAX_PROBES if has_syntax_labels(submission) else "")
            + (POS_PROBES if has_pos_labels(submission) else "")
            + wordnet.format_for_reviewer(submission.top_examples, wordnet.lexicon())
            + (f"{context}\n" if context else "")
            + (f"{related}\n" if related else "")
            + (f"{art}\n" if (art := prior_art.format_for_reviewer(submission.prior_art)) else "")
            + count_line(submission, n, *split(n))
            + LENGTH_MATCHED
        )

    @staticmethod
    def token_check_prompt(
        submission: NeuronAuditSubmission, probes: Sequence[AdversarialProbe], tokenizer
    ) -> str:
        """Return a second-pass prompt for the reviewer: its probes as the model's tokens.

        Then the reviewer can fix labels that it gave by reading words. It shows no
        activations: only the tokenizer runs. For a head, each probe also lists the token
        runs that repeat an earlier one.
        """
        is_head = submission.identifier.kind == "attn_head"
        tagged = has_pos_labels(submission)
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
            if tagged:
                lines.append(pos_line(probe.text))
            blocks.append("\n".join(lines))
        repeats = (
            "Each 'repeated token runs' entry is a longest run of tokens that also occurs "
            "earlier in the probe, with its length in tokens.\n"
            if is_head
            else ""
        )
        unequal = length_warning(probes, tokenizer, bool(submission.first_token_floor))
        rebalance = f"{unequal} Rewrite some probes to match the lengths. " if unequal else ""
        return (
            ReviewerAgent.probe_prompt(submission, len(probes))
            + "\n\nYou wrote the probes below. Here they are as the model's tokens. "
            + repeats
            + (POS_CHECK if tagged else "")
            + rebalance
            + "Check each label against what the hypothesis predicts for these tokens, not "
            "for the words. Where a label is wrong, rewrite the probe so it tests what you "
            "meant (preferred, to keep the share of hard negatives), or relabel it. Leave "
            "the other probes unchanged and write the whole probe set again.\n\n"
            + fence("\n\n".join(blocks))
        )

    def review(
        self,
        submission: NeuronAuditSubmission,
        runner: NeuronHookRunner,
        *,
        probes: Sequence[AdversarialProbe] | None = None,
        policy: ReviewPolicy = DEFAULT_POLICY,
    ) -> VerificationVerdict:
        """Get probes (unless given), measure them and return the verdict."""
        if probes is None:
            probes = self.generate_probes(submission).probes
        return evaluate_submission(submission, runner, probes, self.model, policy)


def repeated_runs(ids: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs ``ids[a:b]`` that also occur ending earlier in ``ids``, as ``(a, b)``.

    ``match[j]`` is the longest run ending at ``j`` that ends at some earlier position
    too; a run is reported where it stops growing.
    """
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


def evaluate_submission(  # long-function: the review's steps, one or two lines each
    submission: NeuronAuditSubmission,
    runner: NeuronHookRunner,
    probes: Sequence[AdversarialProbe],
    reviewer_model: str,
    policy: ReviewPolicy = DEFAULT_POLICY,
    *,
    wire: bool = True,
) -> VerificationVerdict:
    """Measure ``probes`` and the submission's own claims, and decide.

    ``wire=False`` skips the activation wiring (reported, not gated), for example when
    reproducing a record.
    """
    lap = costs.laps()
    submission, unit, scale = settle_first_token_floor(submission, runner, policy)
    threshold = policy.activation_threshold * scale
    inverse = submission.claim == "inverse"
    results, measured_probes, actual = measure_probes(runner, unit, probes, scale, threshold,
                                                      inverse)  # fmt: skip
    labels = [(p.expected_active, p.rival_expected_active or None) for p in measured_probes]
    rivals = compare(submission.natural_language_explanation, submission.rival_explanations,
                     labels, actual)  # fmt: skip
    results += measure_counterfactuals(runner, unit, submission, scale, policy)
    results.append(lookahead_result(runner, unit, submission))
    lap("probe_measurement")
    metrics = review_metrics(
        results,
        policy,
        independent=not same_agent(reviewer_model, submission.submitting_agent_model),
        baseline=confound_baseline(submission, runner, measured_probes, actual, threshold, policy),
        active_share=scan_active_share(submission, policy),
        first_token_floor=submission.first_token_floor,
        inverse=inverse,
    )
    lap("gates")
    causal, joint = causal_readouts(submission, runner, probes, policy, scale)
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
        rivals=rivals,
        inverse_rule_version=INVERSE_RULE_VERSION if inverse else None,
        notes=_summary(results, metrics),
    )


def _probe_score(
    probes: Sequence[AdversarialProbe],
    predicted: Sequence[bool],
    actual: Sequence[bool],
    policy: ReviewPolicy,
) -> float:
    """Return the balanced accuracy of ``predicted`` vs ``actual`` activity.

    It is split by the hypothesis's labels (hard negatives / holdouts), with near-duplicate
    probes clustered.
    """

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
    """Return the best trivial explanation of the probes' activity: (name, accuracy).

    The candidates are the token-identity and position rules, and "never active". Since
    protocol 0.9 they also include each "at least n tokens" rule of probe length (#3525).
    ``None`` when the submission has no confound profile (scan predates protocol 0.3).
    """
    if submission.confounds is None or runner.tokenizer is None or not probes:
        return None
    texts = [p.text for p in probes]
    skip_first = bool(submission.first_token_floor)
    candidates = baseline_predictions(
        submission.confounds, runner.tokenizer, texts, threshold, skip_first
    )
    candidates |= length_predictions(runner.tokenizer, texts, skip_first)  # protocol 0.9
    candidates["never active"] = [False] * len(probes)
    return best_baseline(candidates, probes, actual, policy)


def best_baseline(
    candidates: dict[str, Sequence[bool]],
    probes: Sequence[AdversarialProbe],
    actual: Sequence[bool],
    policy: ReviewPolicy,
) -> tuple[str, float]:
    """The candidate rule that explains the probes' activity best: (name, balanced accuracy).

    Ties go to the first name in sorted order.
    """
    scored = {name: _probe_score(probes, pred, actual, policy) for name, pred in candidates.items()}
    best = max(sorted(scored), key=lambda name: scored[name])
    return best, round(scored[best], 6)


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
    inverse: bool = False,
) -> ReviewMetrics:
    """The rates and gates of the decision rule, from the measured results (docs/PROTOCOL.md §4)."""

    def counts(kind: str):
        rows = [r for r in results if r.kind == kind and r.measured]
        return effective_counts(
            [r.prompt for r in rows], [r.passed for r in rows], policy.probe_cluster_threshold
        )

    negatives, holdouts = counts("hard_negative"), counts("positive_holdout")
    reproduction = _pass_rate(results, {"counterfactual"})
    lookahead = all(r.passed for r in results if r.kind == "lookahead_consistency")
    balanced = balanced_accuracy(negatives, holdouts)
    # The baseline's accuracy is stored rounded, so compare at the same precision. A tie
    # passes (protocol 0.8). Before 0.8, 10/11 lost to a rounded-up 10/11.
    beats = not policy.require_confound_control or (
        baseline is not None and round(balanced, 6) >= baseline[1]
    )

    gates = claim_gates(policy, negatives, holdouts, reproduction, lookahead, independent, beats,
                        active_share, inverse)  # fmt: skip
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
        AdversarialProbe(
            text=r.prompt,
            expected_active=bool(r.expected_active),
            rationale=r.detail,
            rival_expected_active=r.rival_expected_active or [],
        )
        for r in verdict.adversarial_test_results
        if r.kind in {"hard_negative", "positive_holdout"}
    ]
