"""Answer rule 1.1: reviewer corrupted prompts, a drawn seed, a quantile and margins (#4892)."""

import pytest

from openinterpretability.core import answer_records as ar
from tests.test_answer_records import _claim, _measures, _probes, _record, gpt2, means  # noqa: F401

# " sky" on pythia-70m: three of eight random sets moved the answer by 11%, 20% and 27%.
# Negative sets offset them, so the mean (8.3% there) passed rule 1.0's line of 10%.
SKY = [0.11, 0.20, 0.27, -0.10, -0.20, -0.15, 0.0, 0.05]


def test_the_route_must_carry_the_answer_from_each_reviewer_prompt():
    status, gates = ar.decide(_measures(reviewer_patching={"aycd": 0.7, "azcd": 0.2}),
                              ar.AnswerPolicy())  # fmt: skip
    assert status == "REJECTED" and not gates["patching"]


def test_fewer_than_two_reviewer_prompts_fail_the_patching_gates():
    measures = _measures(reviewer_patching={"aycd": 0.7},
                         reviewer_null_patching={"aycd": [0.0]})  # fmt: skip
    gates = ar.decide(measures, ar.AnswerPolicy())[1]
    assert not gates["patching"] and not gates["patching_null"]


def test_a_quantile_catches_large_random_sets_that_a_mean_lets_through():
    sky = _measures(null_patching=SKY)
    assert sum(SKY) / len(SKY) <= 0.1  # the mean passes
    assert ar.quantile([abs(s) for s in SKY], 0.75) > 0.1
    assert not ar.decide(sky, ar.AnswerPolicy())[1]["patching_null"]
    # The same on a reviewer's corrupted prompt.
    theirs = _measures(reviewer_null_patching={"aycd": SKY, "azcd": [0.0, 0.0]})
    assert not ar.decide(theirs, ar.AnswerPolicy())[1]["patching_null"]


def test_a_record_judged_by_rule_1_0_keeps_its_verdict():
    old = ar.AnswerPolicy(protocol_version="1.0")
    measures = _measures(null_patching=SKY, reviewer_patching={}, reviewer_null_patching={},
                         null_seed=None)  # fmt: skip
    status, gates = ar.decide(measures, old)
    assert status == "VERIFIED" and all(gates.values())
    assert ar.decide(measures, ar.AnswerPolicy())[0] == "REJECTED"


def test_quantile_interpolates_as_numpy_does():
    numpy = pytest.importorskip("numpy")
    values = [0.3, 0.1, 0.2, 0.05, 0.4]
    assert ar.quantile(values, 0.75) == pytest.approx(float(numpy.quantile(values, 0.75)))
    assert ar.quantile([0.5], 0.75) == 0.5


def test_margins_show_how_narrowly_each_gate_passed():
    narrow = _measures(sufficiency=0.536, null_patching=[0.0, 0.083])
    rows = ar.margins(narrow, ar.AnswerPolicy())
    assert rows["sufficiency"]["margin"] == pytest.approx(0.036)
    assert rows["patching"]["value"] == pytest.approx(0.6)  # the lowest over the prompts
    assert rows["patching_null"]["margin"] == pytest.approx(0.1 - ar.quantile([0.0, 0.083], 0.75))
    assert rows["beats_null"]["value"] == pytest.approx(3.0 / 0.15)
    failing = ar.margins(_measures(patching=0.2), ar.AnswerPolicy())["patching"]
    assert failing["margin"] == pytest.approx(-0.3)


def test_the_reviewers_corrupted_prompts_are_checked_before_any_measure():
    from types import SimpleNamespace

    from openinterpretability.answer_review import check_probes

    tokenizer = SimpleNamespace(encode=lambda text: list(text))  # one token per character
    policy, claim = ar.AnswerPolicy(), _claim()
    assert check_probes(tokenizer, claim, _probes(), policy) == []
    bad = _probes().model_copy(update={"corrupted": ["axcd", "too long"]})
    found = " ".join(check_probes(tokenizer, claim, bad, policy))
    assert "repeats the claim's" in found and "is not 4 tokens" in found
    few = _probes().model_copy(update={"corrupted": ["aycd"]})
    assert "wrote 1 corrupted prompts" in " ".join(check_probes(tokenizer, claim, few, policy))
    assert check_probes(tokenizer, claim, few, ar.AnswerPolicy(protocol_version="1.0")) == []


def test_the_review_draws_the_seed_and_measures_each_reviewer_prompt(gpt2, means, monkeypatch):  # noqa: F811
    from openinterpretability import answer_review

    monkeypatch.setattr(answer_review, "draw_seed", lambda: 1234)
    record = answer_review.review(gpt2, means, _claim(), _probes(), ar.AnswerPolicy(null_sets=3))
    m = record.verdict.measures
    assert m.null_seed == 1234
    assert set(m.reviewer_patching) == {"aycd", "azcd"}
    assert all(len(v) == 3 for v in m.reviewer_null_patching.values())
    given = answer_review.review(gpt2, means, _claim(), _probes(), ar.AnswerPolicy(null_sets=3),
                                 seed=5)  # fmt: skip
    assert given.verdict.measures.null_seed == 5


def test_the_page_shows_margins_reviewer_prompts_and_the_seed(tmp_path):
    from openinterpretability.builder.explanation_pages import AnswerView

    view = AnswerView(_record(), "wiki/answers/x.json")
    assert set(view.margins) == set(view.record.verdict.gates)
    assert view.null_levels == {
        "aycd": pytest.approx(ar.quantile([0.0, 0.02], 0.75)),
        "azcd": pytest.approx(ar.quantile([0.01, 0.0], 0.75)),
    }
