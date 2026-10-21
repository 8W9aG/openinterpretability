import json
import math
import statistics
import threading
from types import SimpleNamespace

import pytest
import torch

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
from openinterp.harness.causal import (
    _matched_units,
    _percentile,
    causal_metrics,
    claimed_token_weights,
)
from openinterp.harness.hooks import top_activating_examples
from openinterp.harness.jlens import vocabulary_lens

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

    lens = vocabulary_lens(toy_runner, ["hello"], 1, 2, "positive", k=3)
    assert agent.propose(ident, examples, lens) is draft
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


def test_structured_call_records_usage_across_retries(draft):
    from openinterp.core import costs

    class Client:  # instructor's hooks: one raw LiteLLM response per attempt
        def __init__(self):
            self.hooks = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def on(self, name, fn):
            self.hooks.append(fn)

        def off(self, name, fn):
            self.hooks.remove(fn)

        def create(self, **kwargs):
            for cost in (0.01, 0.02):
                usage = SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=10,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=40),
                )
                response = SimpleNamespace(usage=usage, _hidden_params={"response_cost": cost})
                for hook in self.hooks:
                    hook(response)
            return draft

    costs.reset()
    client = Client()
    ProposerAgent("anthropic/m", client=client).structured(HypothesisDraft, "go")
    (call,) = costs.measured().llm_calls
    assert (call.agent, call.model, call.attempts) == ("proposer", "anthropic/m", 2)
    assert (call.prompt_tokens, call.completion_tokens, call.cached_tokens) == (200, 20, 80)
    assert call.cost_usd == pytest.approx(0.03)
    assert call.prompt_chars == len(ProposerAgent.system_prompt) + 2
    assert [s.stage for s in costs.measured().stages] == ["proposer"]
    assert client.hooks == []  # unregistered after the call


def test_build_submission_measures_evidence(ident, draft, toy_runner):
    examples, values = top_activating_examples(toy_runner, TEXTS, 1, 2, "positive", k=3)
    assert examples[0].max_activation >= examples[-1].max_activation
    sub = ProposerAgent("openai/test").build_submission(
        ident, draft, runner=toy_runner, examples=examples
    )
    assert sub.max_activation == examples[0].max_activation
    assert all(cf.observed_delta is not None for cf in sub.counterfactuals)
    assert sub.model_num_layers == 3
    assert sub.model_mlp_width == toy_runner.mlp_width(ident.layer)
    assert sub.lookahead.kl_divergence >= 0


def test_probe_set_bounds():
    probe = AdversarialProbe(text="x", expected_active=False, rationale="r")
    with pytest.raises(ValueError):
        ProbeSet(probes=[probe] * 9)
    with pytest.raises(ValueError):
        ProbeSet(probes=[probe] * 21)


def _submission(ident, toy_runner, draft, *, honest: bool):
    from openinterp.harness.scan import scan_neuron

    scan = scan_neuron(toy_runner, TEXTS, 1, 2, "positive")
    sub = ProposerAgent("openai/test").build_submission(
        ident,
        draft,
        runner=toy_runner,
        examples=scan.examples,
        confounds=scan.confounds,
        correlated=scan.correlated,
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


def test_same_model_review_cannot_verify_when_required(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, ident, sub, honest=True)
    policy = SMALL.model_copy(update={"require_independent_review": True})
    verdict = evaluate_submission(sub, toy_runner, probes, sub.submitting_agent_model, policy)
    assert verdict.metrics.gates["independent_review"] is False
    assert verdict.status != "VERIFIED"
    assert "independent_review" in verdict.notes


def test_same_model_review_is_recorded_not_gated_by_default(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, ident, sub, honest=True)
    verdict = evaluate_submission(sub, toy_runner, probes, sub.submitting_agent_model, SMALL)
    assert verdict.metrics.independent_review is False
    assert verdict.metrics.gates["independent_review"] is True


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
    serial = iter(range(10_000))

    def r(kind, passed, n):
        # Distinct word sets, so no two probes cluster as near-duplicates.
        return [
            AdversarialTestResult(kind=kind, prompt=f"w{i}a w{i}b w{i}c", passed=passed)
            for i in (next(serial) for _ in range(n))
        ]

    return (
        r("hard_negative", True, neg_pass)
        + r("hard_negative", False, neg_fail)
        + r("positive_holdout", True, pos_pass)
        + r("positive_holdout", False, pos_fail)
        + r("counterfactual", True, cf_pass)
        + r("counterfactual", False, cf_fail)
        + r("lookahead_consistency", True, 1)
    )


BASELINE = ("never active", 0.5)


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
    metrics = review_metrics(_results(*counts), policy, independent=True, baseline=BASELINE)
    assert decide(metrics, policy) == status


def test_balanced_accuracy_ignores_class_imbalance():
    metrics = review_metrics(
        _results(10, 0, 2, 3), ReviewPolicy(), independent=True, baseline=BASELINE
    )
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


def test_near_duplicate_probes_count_once():
    dupes = [
        AdversarialTestResult(
            kind="hard_negative", prompt=f"Many cities including Paris {i}", passed=True
        )
        for i in range(8)
    ]
    metrics = review_metrics(
        dupes + _results(0, 0, 5, 0), ReviewPolicy(), independent=True, baseline=BASELINE
    )
    assert metrics.effective_hard_negatives == 1
    assert metrics.gates["enough_hard_negatives"] is False


def test_confound_gate():
    results = _results(10, 0, 4, 1)  # balanced accuracy 0.9
    policy = ReviewPolicy()
    assert review_metrics(
        results, policy, independent=True, baseline=("token identity", 0.9)
    ).gates["beats_confound_baseline"]  # ties pass
    beaten = review_metrics(results, policy, independent=True, baseline=("token identity", 0.95))
    assert beaten.gates["beats_confound_baseline"] is False
    assert beaten.confound_baseline == "token identity"
    missing = review_metrics(results, policy, independent=True, baseline=None)
    assert missing.gates["beats_confound_baseline"] is False  # no confound profile


def test_review_records_confounds_and_joint_ablation(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    assert sub.confounds is not None and sub.correlated_neurons
    probes = _probes(toy_runner, ident, sub, honest=True)
    policy = ReviewPolicy(min_hard_negatives=1, min_holdouts=1, min_partner_correlation=0.0)
    verdict = evaluate_submission(sub, toy_runner, probes, "anthropic/reviewer", policy)
    assert verdict.metrics.confound_baseline_accuracy is not None
    ja = verdict.joint_ablation
    assert ja is not None and len(ja.partners) == 3
    assert ja.kl_joint >= 0 and ja.kl_single >= 0


def test_external_agent_reads_response_file(tmp_path):
    agent = ProposerAgent("external/test-llm", exchange_dir=tmp_path, poll_interval=0.01)
    pair = {"base_prompt": "a", "counterfactual_prompt": "b", "expected_delta": 0.5}
    answer = {
        "explanation": "fires on x",
        "activating_tokens": ["x"],
        "mechanism": "retrospective",
        "counterfactuals": [dict(pair, rationale=str(i)) for i in range(3)],
    }

    def respond():
        prompt = tmp_path / "ProposerAgent.prompt.md"
        while not prompt.exists():
            threading.Event().wait(0.01)
        assert "counterfactuals" in prompt.read_text()  # schema is included
        (tmp_path / "ProposerAgent.response.json").write_text('{"explanation": 1}')
        while not (tmp_path / "ProposerAgent.error.txt").exists():
            threading.Event().wait(0.01)
        (tmp_path / "ProposerAgent.response.json").write_text(json.dumps(answer))

    threading.Thread(target=respond, daemon=True).start()
    draft = agent.structured(HypothesisDraft, "describe the neuron")
    assert draft.explanation == "fires on x"
    assert not (tmp_path / "ProposerAgent.error.txt").exists()


def test_external_agent_times_out(tmp_path):
    agent = ReviewerAgent("external/x", exchange_dir=tmp_path, timeout=0.05, poll_interval=0.01)
    with pytest.raises(AgentError):
        agent.structured(ProbeSet, "probes")


def test_claimed_token_weights_for_lookahead(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    lookahead = sub.lookahead.model_copy(
        update={"is_lookahead": True, "target_upcoming_tokens": ["a", "b", "a", "too long"]}
    )
    weights, tokens = claimed_token_weights(
        sub.model_copy(update={"lookahead": lookahead}), toy_runner
    )
    assert tokens == ["a", "b"]  # deduplicated; multi-token strings are dropped
    assert float(weights.sum()) == pytest.approx(1.0)
    causal = causal_metrics(
        sub.model_copy(update={"lookahead": lookahead}), toy_runner, null_neurons=5
    )
    assert causal.target == "claimed_tokens" and causal.target_tokens == ["a", "b"]


def test_causal_metrics_rank_against_matched_neurons(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    retro = sub.lookahead.model_copy(update={"is_lookahead": False})
    sub = sub.model_copy(update={"lookahead": retro})
    assert claimed_token_weights(sub, toy_runner) is None
    causal = causal_metrics(sub, toy_runner, null_neurons=5)
    assert causal.target == "next_token_kl" and causal.target_tokens == []
    assert causal.null_neurons == 5
    assert causal.sufficiency_effect >= 0 and causal.necessity_effect >= 0  # KL


def test_causal_metrics_record_each_pair(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    causal = causal_metrics(sub, toy_runner, null_neurons=5)
    assert [p.pair_index for p in causal.pairs] == list(range(causal.pairs_scored))
    suff = [p.sufficiency for p in causal.pairs]
    assert causal.sufficiency_score == pytest.approx(
        statistics.median(p.percentile for p in suff), abs=1e-6
    )
    assert causal.sufficiency_effect == pytest.approx(statistics.median(p.effect for p in suff))
    assert all(0 <= p.null_median <= p.null_max for p in suff)


def test_causal_metrics_skip_control_pairs(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    all_pairs = causal_metrics(sub, toy_runner, null_neurons=5)
    assert all_pairs.pairs_scored == len(sub.counterfactuals)
    assert causal_metrics(sub, toy_runner, null_neurons=5, min_change=1e9) is None


def test_matched_neurons_are_closest_in_change():
    change = torch.tensor([0.0, 5.0, 4.0, 0.1, 6.0, 9.0])
    assert _matched_units(change, 1, 3) == [2, 4, 5]


def test_percentile_counts_ties_as_half():
    assert _percentile(2.0, [1.0, 2.0, 3.0, 0.0]) == pytest.approx(0.625)
    assert math.isnan(_percentile(1.0, []))
