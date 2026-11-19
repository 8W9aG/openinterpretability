"""Inverse claims: when a default-on pole goes quiet (inverse_claims.py, #4951)."""

from types import SimpleNamespace

import click
import pytest

from openinterpretability import inverse_claims as ic
from openinterpretability.agents.reviewer import ReviewerAgent, evaluate_submission
from openinterpretability.core.schemas import AuditRecord, ReviewPolicy
from openinterpretability.explain_stopped import inverse_experiments, lines
from openinterpretability.verdict_regression import recompute
from tests.test_agents import SMALL, _probes, _submission
from tests.test_reproduce import draft, ident  # noqa: F401  (fixtures)


def test_an_inverse_claim_swaps_which_probes_are_holdouts():
    assert ic.probe_kind(True, inverse=False) == "positive_holdout"
    assert ic.probe_kind(False, inverse=False) == "hard_negative"
    assert ic.probe_kind(False, inverse=True) == "positive_holdout"  # predicted quiet
    assert ic.probe_kind(True, inverse=True) == "hard_negative"  # predicted to stay on


def test_an_inverse_claim_swaps_the_default_on_gate():
    counts = SimpleNamespace(clusters=10, rate=1.0)
    args = (ReviewPolicy(), counts, counts, 1.0, True, True, True)
    detector = ic.gates(*args, active_share=0.9)
    inverse = ic.gates(*args, active_share=0.9, inverse=True)
    assert detector["not_default_on"] is False and "is_default_on" not in detector
    assert inverse["is_default_on"] is True and "not_default_on" not in inverse
    assert ic.gates(*args, active_share=0.2, inverse=True)["is_default_on"] is False
    assert ic.gates(*args, active_share=None, inverse=True)["is_default_on"] is False


def test_the_scan_takes_an_inverse_claim_only_on_a_default_on_pole(identifier):
    from openinterpretability.evidence import Unauditable

    on, off = [1.0] * 9 + [0.0], [1.0] + [0.0] * 9
    with pytest.raises(Unauditable):
        ic.check_scan(identifier, on, 1.0, None, inverse=False)  # a detector stops
    ic.check_scan(identifier, on, 1.0, None, inverse=True)  # an inverse claim goes on
    ic.check_scan(identifier, off, 1.0, None, inverse=False)
    with pytest.raises(click.UsageError, match="not default-on"):
        ic.check_scan(identifier, off, 1.0, None, inverse=True)
    head = SimpleNamespace(kind="attn_head")
    with pytest.raises(click.UsageError, match="neuron poles only"):
        ic.check_scan(head, on, 1.0, None, inverse=True)


def test_a_review_of_an_inverse_claim_records_its_rule(ident, toy_runner, draft):  # noqa: F811
    sub = _submission(ident, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, ident, sub, honest=True)
    detector = evaluate_submission(sub, toy_runner, probes, "external/r", SMALL, wire=False)
    inverse_sub = sub.model_copy(update={"claim": "inverse"})
    inverse = evaluate_submission(inverse_sub, toy_runner, probes, "external/r", SMALL, wire=False)
    kinds = {r.prompt: r.kind for r in detector.adversarial_test_results}
    swapped = {r.prompt: r.kind for r in inverse.adversarial_test_results}
    flip = {"hard_negative": "positive_holdout", "positive_holdout": "hard_negative"}
    assert all(swapped[p] == flip[k] for p, k in kinds.items() if k in flip)
    assert inverse.inverse_rule_version == ic.INVERSE_RULE_VERSION
    assert detector.inverse_rule_version is None
    assert "is_default_on" in inverse.metrics.gates
    # The regression check recomputes an inverse record's gates the same way.
    record = AuditRecord(submission=inverse_sub, verdict=inverse)
    metrics, status = recompute(record)
    assert metrics.gates == inverse.metrics.gates and status == inverse.status


def test_the_reviewer_is_asked_for_quiet_holdouts(ident, toy_runner, draft):  # noqa: F811
    sub = _submission(ident, toy_runner, draft, honest=True)
    plain = ReviewerAgent.probe_prompt(sub, 30)
    inverse = ReviewerAgent.probe_prompt(sub.model_copy(update={"claim": "inverse"}), 30)
    assert "INVERSE CLAIM" not in plain and "INVERSE CLAIM" in inverse
    assert "positive holdouts that silence it" in inverse


def test_the_proposer_is_told_only_for_an_inverse_scan():
    assert ic.inverse_preamble(SimpleNamespace(claim="detector")) == ""
    assert "QUIET" in ic.inverse_preamble(SimpleNamespace(claim="inverse"))


def test_the_top_default_on_poles_by_gain_get_inverse_experiments():
    rows = [{"target": f"L1/N{i}+", "unit": (1, i, "positive"), "share": 0.01, "gain": g,
             "reason": r}
            for i, (g, r) in enumerate([(0.05, "default_on"), (0.04, "dead"),
                                        (0.03, "default_on"), (0.02, "no follow-up")])]  # fmt: skip
    rows.append({"target": "L2/H1", "unit": (2, 1, "head"), "share": 0.01, "gain": 0.06,
                 "reason": "default_on"})  # fmt: skip
    picks = inverse_experiments(rows, "m/x", cap=5)
    assert [p["target"] for p in picks] == ["L1/N0+", "L1/N2+"]  # neuron poles, default-on
    assert (
        "--inverse" in picks[0]["command"] and "--neuron 0 --pole positive" in picks[0]["command"]
    )
    assert len(inverse_experiments(rows, "m/x", cap=1)) == 1
    text = "\n".join(lines(rows, "m/x"))
    assert "+5.0% coverage" in text and "inverse experiment for L1/N0+" in text
