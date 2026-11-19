"""Rival readings, scored beside the hypothesis on the review's probes (core/rivals.py, #3197)."""

import pytest

from openinterpretability.agents.reviewer import evaluate_submission, probes_from_verdict
from openinterpretability.core.rivals import compare, rival_prompt
from openinterpretability.core.schemas import AuditRecord
from tests.test_agents import SMALL, _probes, _submission
from tests.test_reproduce import draft, ident  # noqa: F401  (fixtures)


def test_the_reading_that_predicts_the_probes_best_survives():
    actual = [True, True, False, False]
    labels = [(True, [True, False]), (False, [True, True]), (False, [False, False]),
              (True, [False, True])]  # fmt: skip
    found = compare("hyp", ["broad", "narrow"], labels, actual)
    hyp, broad, narrow = found.readings
    assert (hyp.balanced_accuracy, broad.balanced_accuracy) == (0.5, 1.0)
    assert narrow.balanced_accuracy == 0.5 and found.survivor == 1
    assert (broad.disagreements, narrow.disagreements) == (2, 2)
    assert all(r.probes == 4 for r in found.readings)


def test_ties_go_to_the_hypothesis_and_no_rivals_is_no_comparison():
    labels = [(True, [True]), (False, [False])]
    assert compare("hyp", ["same"], labels, [True, False]).survivor == 0
    assert compare("hyp", [], labels, [True, False]) is None


def test_a_probe_without_matching_rival_labels_counts_for_the_hypothesis_only():
    labels = [(True, [True, True]), (False, None), (False, [False])]  # the last has too few
    found = compare("hyp", ["a", "b"], labels, [True, False, False])
    assert found.readings[0].probes == 3 and found.readings[1].probes == 1


def test_the_reviewer_is_asked_for_rival_labels_only_when_there_are_rivals():
    assert rival_prompt([], 30) == ""
    text = rival_prompt(["Fires on any vowel."], 30)
    assert "rival_expected_active" in text and "Fires on any vowel." in text
    assert "at least 6 probes" in text


@pytest.fixture
def reviewed(ident, toy_runner, draft):  # noqa: F811
    """A review of a submission with one rival that predicts the opposite of the hypothesis."""
    with_rival = draft.model_copy(update={"rivals": ["Fires on anything but repeats."]})
    sub = _submission(ident, toy_runner, with_rival, honest=True)
    probes = [
        p.model_copy(update={"rival_expected_active": [not p.expected_active]})
        for p in _probes(toy_runner, ident, sub, honest=True)
    ]
    return sub, evaluate_submission(sub, toy_runner, probes, "external/reviewer", SMALL)


def test_a_review_scores_the_rivals_without_changing_the_verdict(reviewed, toy_runner):
    sub, verdict = reviewed
    assert sub.rival_explanations == ["Fires on anything but repeats."]
    hyp, rival = verdict.rivals.readings
    assert verdict.rivals.survivor == 0 and hyp.balanced_accuracy > rival.balanced_accuracy
    assert rival.disagreements == rival.probes > 0
    plain = evaluate_submission(
        sub.model_copy(update={"rival_explanations": []}), toy_runner,
        probes_from_verdict(verdict), "external/reviewer", SMALL,
    )  # fmt: skip
    assert plain.rivals is None and (plain.status, plain.metrics) == (
        verdict.status,
        verdict.metrics,
    )
    # The labels are stored, so reproduce re-derives the comparison from the record.
    again = probes_from_verdict(verdict)
    assert all(len(p.rival_expected_active) == 1 for p in again)
    redone = evaluate_submission(sub, toy_runner, again, "external/reviewer", SMALL)
    assert redone.rivals == verdict.rivals


def test_the_page_shows_each_reading_and_the_best(reviewed, tmp_path):
    from openinterpretability.builder import compile_site

    sub, verdict = reviewed
    compile_site([AuditRecord(submission=sub, verdict=verdict)], tmp_path)
    # The Changelog page can name the feature too, so look only at experiment pages.
    pages = [p for p in tmp_path.rglob("*.html") if "changelog" not in p.as_posix().lower()]
    (page,) = [p for p in pages if "Rival readings" in p.read_text()]
    html = page.read_text()
    assert "Fires on anything but repeats." in html and ">best</span>" in html
