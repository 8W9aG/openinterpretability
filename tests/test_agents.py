import pytest

from openinterp.agents.base import LLMAgent
from openinterp.agents.proposer import CounterfactualDraft, HypothesisDraft, ProposerAgent
from openinterp.agents.reviewer import (
    AdversarialProbe,
    ProbeSet,
    ReviewerAgent,
    ReviewPolicy,
    _delta_agrees,
    decide,
    evaluate_submission,
    probes_from_verdict,
    review_metrics,
)
from openinterp.core.exceptions import AgentError
from openinterp.core.schemas import AdversarialTestResult, CounterfactualPair, NeuronIdentifier
from openinterp.harness.hooks import top_activating_examples

# The toy model's probe set is small, so relax only the minimum-count gates.
SMALL = ReviewPolicy(min_hard_negatives=1, min_holdouts=1)
TEXTS = ["abcde", "hello world", "zzzz", "quoted 'text'", "mixed Case 123", "aaaa bbbb"]


@pytest.fixture
def ident():
    return NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")


@pytest.fixture
def draft():
    return HypothesisDraft(
        explanation="Fires on repeated letters.",
        activating_tokens=["a"],
        mechanism="retrospective",
        counterfactuals=[
            CounterfactualDraft(
                base_prompt=b, counterfactual_prompt=c, expected_delta=0.5, rationale="r"
            )
            for b, c in [("aaaa", "abcd"), ("zzzz", "zyxw"), ("bbbb", "bcde")]
        ],
    )


def fake_client(mocker, response):
    client = mocker.Mock()
    client.chat.completions.create.return_value = response
    return client


def test_structured_call_passes_model_and_messages(mocker, draft, ident, toy_runner):
    client = fake_client(mocker, draft)
    agent = ProposerAgent("openai/test-model", client=client)
    examples, _ = top_activating_examples(toy_runner, TEXTS, 1, 2, "positive", k=3)

    assert agent.propose(ident, examples, ["x"]) is draft
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/test-model"
    assert kwargs["response_model"] is HypothesisDraft
    assert "temperature" not in kwargs
    assert kwargs["messages"][0]["role"] == "system"
    assert "[[" in kwargs["messages"][1]["content"]  # peak token marked


def test_agent_errors_are_wrapped(mocker):
    client = mocker.Mock()
    client.chat.completions.create.side_effect = RuntimeError("rate limited")

    class Dummy(LLMAgent):
        pass

    with pytest.raises(AgentError, match="rate limited"):
        Dummy("m", client=client).structured(ProbeSet, "hi")


def test_build_submission_measures_evidence(ident, draft, toy_runner):
    examples, values = top_activating_examples(toy_runner, TEXTS, 1, 2, "positive", k=3)
    assert examples[0].max_activation >= examples[-1].max_activation
    sub = ProposerAgent("openai/test").build_submission(
        ident, draft, runner=toy_runner, examples=examples
    )
    assert sub.max_activation == examples[0].max_activation
    assert all(cf.observed_delta is not None for cf in sub.counterfactuals)
    assert sub.model_num_layers == 3
    assert sub.lookahead.kl_divergence >= 0


def test_probe_set_bounds():
    probe = AdversarialProbe(text="x", expected_active=False, rationale="r")
    with pytest.raises(ValueError):
        ProbeSet(probes=[probe] * 9)
    with pytest.raises(ValueError):
        ProbeSet(probes=[probe] * 21)


def _submission(ident, toy_runner, draft, *, honest: bool):
    examples, _ = top_activating_examples(toy_runner, TEXTS, 1, 2, "positive")
    sub = ProposerAgent("openai/test").build_submission(
        ident, draft, runner=toy_runner, examples=examples
    )
    # Honest submissions predict exactly what was measured; dishonest ones the opposite.
    pairs = [
        CounterfactualPair(
            base_prompt=cf.base_prompt,
            counterfactual_prompt=cf.counterfactual_prompt,
            expected_delta=cf.observed_delta if honest else -cf.observed_delta - 0.5,
        )
        for cf in sub.counterfactuals
    ]
    return sub.model_copy(update={"counterfactuals": pairs})


def _probes(toy_runner, ident, sub, *, honest: bool):
    threshold = 0.25 * sub.max_activation
    probes = []
    for text in ["a", "ab", "ba", "zz", "q", "hello", "world", "123", "xyz", "aaaa bbbb", "Case"]:
        active = (
            toy_runner.max_pole_strength(text, ident.layer, ident.neuron_idx, ident.pole)
            >= threshold
        )
        probes.append(
            AdversarialProbe(
                text=text, expected_active=active if honest else not active, rationale="r"
            )
        )
    return probes


def test_review_verifies_accurate_hypothesis(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, ident, sub, honest=True)
    # Lookahead claim matches whatever is measured, so every component passes.
    from openinterp.harness.lookahead import measure_lookahead

    measured = measure_lookahead(
        toy_runner, [c.base_prompt for c in sub.counterfactuals], 1, 2, "positive"
    )
    sub = sub.model_copy(
        update={
            "lookahead": sub.lookahead.model_copy(update={"is_lookahead": measured.is_lookahead})
        }
    )

    verdict = evaluate_submission(sub, toy_runner, probes, "anthropic/reviewer", SMALL)
    assert verdict.status == "VERIFIED", verdict.notes
    assert verdict.policy == SMALL
    assert verdict.confidence_score == pytest.approx(1.0)
    assert verdict.causal_metrics is not None and verdict.causal_metrics.hydra_mitigated
    kinds = {r.kind for r in verdict.adversarial_test_results}
    assert {"counterfactual", "lookahead_consistency"} <= kinds
    assert verdict.timestamp.tzinfo is not None


def test_review_rejects_wrong_hypothesis(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=False)
    probes = _probes(toy_runner, ident, sub, honest=False)
    verdict = evaluate_submission(sub, toy_runner, probes, "anthropic/reviewer", SMALL)
    assert verdict.status == "REJECTED"
    assert verdict.confidence_score < SMALL.reject_below_balanced_accuracy


def test_same_model_review_cannot_verify(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, ident, sub, honest=True)
    verdict = evaluate_submission(sub, toy_runner, probes, sub.submitting_agent_model, SMALL)
    assert verdict.metrics.gates["independent_review"] is False
    assert verdict.status != "VERIFIED"
    assert "independent_review" in verdict.notes


def test_probes_round_trip_through_verdict(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, ident, sub, honest=True)
    verdict = evaluate_submission(sub, toy_runner, probes, "anthropic/reviewer", SMALL)
    assert [(p.text, p.expected_active) for p in probes_from_verdict(verdict)] == [
        (p.text, p.expected_active) for p in probes
    ]


@pytest.mark.parametrize(
    ("expected", "observed", "ok"),
    [
        (0.40, 0.31, True),  # within 50% of the prediction
        (0.40, 0.11, False),  # right sign, a quarter of the size: no longer passes
        (0.30, 0.08, False),
        (0.00, 0.09, True),  # control pair inside the absolute floor
        (0.00, 0.11, False),
        (0.40, -0.05, False),
        (-0.60, -0.40, True),
    ],
)
def test_relative_delta_tolerance(expected, observed, ok):
    assert _delta_agrees(expected, observed, ReviewPolicy()) is ok


def _results(neg_pass, neg_fail, pos_pass, pos_fail, cf_pass=4, cf_fail=0):
    def r(kind, passed):
        return AdversarialTestResult(kind=kind, prompt="p", passed=passed)

    return (
        [r("hard_negative", True)] * neg_pass
        + [r("hard_negative", False)] * neg_fail
        + [r("positive_holdout", True)] * pos_pass
        + [r("positive_holdout", False)] * pos_fail
        + [r("counterfactual", True)] * cf_pass
        + [r("counterfactual", False)] * cf_fail
        + [r("lookahead_consistency", True)]
    )


@pytest.mark.parametrize(
    ("counts", "status"),
    [
        ((10, 0, 5, 0), "VERIFIED"),
        ((10, 0, 2, 3), "NEEDS_REVISION"),  # 10 easy negatives can't hide failed holdouts
        ((10, 0, 0, 5), "REJECTED"),  # balanced accuracy 0.5 = chance
        ((2, 0, 2, 0), "NEEDS_REVISION"),  # too few probes to judge either way
        ((10, 0, 5, 0, 1, 3), "NEEDS_REVISION"),  # counterfactuals don't reproduce
    ],
)
def test_decision_rule(counts, status):
    policy = ReviewPolicy()
    metrics = review_metrics(_results(*counts), policy, independent=True)
    assert decide(metrics, policy) == status


def test_balanced_accuracy_ignores_class_imbalance():
    metrics = review_metrics(_results(10, 0, 2, 3), ReviewPolicy(), independent=True)
    assert metrics.specificity == 1.0 and metrics.sensitivity == 0.4
    assert metrics.balanced_accuracy == pytest.approx(0.7)


def test_reviewer_agent_uses_llm_probes(mocker, ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    probe_set = ProbeSet(probes=_probes(toy_runner, ident, sub, honest=True))
    client = fake_client(mocker, probe_set)
    verdict = ReviewerAgent("anthropic/reviewer", client=client).review(
        sub, toy_runner, policy=SMALL
    )
    assert client.chat.completions.create.call_args.kwargs["response_model"] is ProbeSet
    assert verdict.reviewer_agent_model == "anthropic/reviewer"
    assert (
        sum(
            r.kind in {"hard_negative", "positive_holdout"}
            for r in verdict.adversarial_test_results
        )
        == 11
    )
