"""Proposing a circuit by attribution, and reviewing one too big to measure member by member."""

from openinterpretability import circuit_review
from openinterpretability.circuit_review import propose, review
from openinterpretability.core import circuit_records as cr
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import FakeTokenizer

PAIR = cr.PromptPair(clean="abcd", corrupted="xbcd", good="e", bad="f")
TASK = cr.CircuitTask(name="t", description="d", pairs=[PAIR])
BASE = dict(model_name="toy/model", circuit_id="C1", explanation="e",
            submitting_agent_model="external/a")  # fmt: skip


def test_a_proposal_grows_the_cluster_to_its_size(tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    cluster = ["models/m/L0/N1.html", "not a member"]
    hypothesis = propose(runner, BASE, TASK, cluster, size=3)
    assert hypothesis.members[0] == "L0/N1" and len(hypothesis.members) == 3
    assert len(set(hypothesis.members)) == 3
    assert hypothesis.cluster_members == sorted(cluster)
    assert hypothesis.task == TASK and hypothesis.circuit_id == "C1"


def test_a_big_circuit_is_reviewed_without_each_members_necessity(tiny_gpt2, monkeypatch):
    monkeypatch.setattr(circuit_review, "MAX_MEMBERS_MEASURED", 1)
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    hypothesis = cr.CircuitHypothesis(members=["L0/N1", "L1/H0"], task=TASK, **BASE)
    probes = cr.CircuitProbes(
        holdouts=[PAIR], reviewer_agent_model="external/b", negative_tasks=[TASK]
    )
    record = review(runner, hypothesis, probes, cr.CircuitPolicy(null_sets=2))
    measures = record.verdict.measures
    assert measures.member_necessities == {} and len(measures.null_necessities) == 2
    assert cr.decide(measures, record.verdict.policy)[0] == record.verdict.status
