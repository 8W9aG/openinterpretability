"""A pole's standing and the reviewer agreement (builder/standing.py)."""

from datetime import timedelta

import pytest

from openinterpretability.builder import standing as st
from openinterpretability.builder.standing import contested_units, reviewer_agreement, score
from openinterpretability.builder.tree import build_tree
from openinterpretability.core.errata import record_ref
from openinterpretability.core.explanation import ExplanationScores, JudgeScore
from openinterpretability.core.schemas import ReviewMetrics
from tests.test_compiler import (  # noqa: F401  (records is a fixture)
    _reviewed_by,
    _rival,
    _self_reviewed,
    _standing,
    experiment,
    records,
)
from tests.test_errata import erratum, later


def _detection(rec, balanced_accuracy):
    """``rec`` with a held-out detection score."""
    scores = ExplanationScores(
        slug=rec.submission.identifier.slug, explanation="x", threshold=0.6, window_tokens=32,
        quantiles=1, quantile_edges=[], seed=3, corpus="c", candidate_windows=10,
        active_windows=5,
        detection=JudgeScore(balanced_accuracy=balanced_accuracy, true_positive_rate=None,
                             false_positive_rate=None, recall_by_quantile=[]),
    )  # fmt: skip
    return rec.model_copy(
        update={"verdict": rec.verdict.model_copy(update={"explanation_scores": scores})}
    )


def test_more_rejected_than_verified_readings_make_it_rejected(records):  # noqa: F811
    verified = records[1]
    for reviewer, hours in (("deepseek/flash", 3), ("muse/spark", 4)):
        records.append(
            _reviewed_by(verified, reviewer, status="REJECTED", hours=hours, counts=(2, 8, 0, 5))
        )
    standing, unit = _standing(records)
    assert standing.status == "REJECTED" and not unit.verified
    assert [e.record.verdict.status for e in standing.readings] == [
        "REJECTED", "REJECTED", "VERIFIED",
    ]  # fmt: skip


def test_an_even_split_goes_to_the_held_out_detection_score(records):  # noqa: F811
    records[1] = _detection(records[1], 0.9)
    rejected = _reviewed_by(records[1], "deepseek/flash", status="REJECTED", hours=3,
                            counts=(2, 8, 0, 5))  # fmt: skip
    records.append(_detection(rejected, 0.6))
    assert _standing(records)[0].status == "VERIFIED"
    records[1] = _detection(records[1], 0.6)
    records[-1] = _detection(rejected, 0.9)
    assert _standing(records)[0].status == "REJECTED"


def test_reviews_by_the_same_agent_are_no_cross_model_pair(records):  # noqa: F811
    verified = records[1]
    same = verified.verdict.reviewer_agent_model
    records.append(_reviewed_by(verified, same, status="VERIFIED", hours=3))
    agreement = reviewer_agreement(build_tree(records, []))
    assert agreement["pairs"] == 1 and agreement["pair_agreement"] == 1.0
    assert agreement["cross_model_pairs"] == 0 and agreement["cross_model_agreement"] is None


@pytest.mark.parametrize(
    ("verified", "rejected", "status"),
    [(0.9, 0.7, "VERIFIED"), (0.6, 0.8, "REJECTED"), (0.80, 0.78, "CONTESTED")],
)
def test_scores_settle_verified_against_rejected(records, monkeypatch, verified, rejected,  # noqa: F811
                                                 status):  # fmt: skip
    """An even VERIFIED–REJECTED split, and a REJECTED rival, go to the score (#4900)."""
    from openinterpretability.builder import standing as st

    table = {"VERIFIED": verified, "REJECTED": rejected}
    monkeypatch.setattr(st, "score", lambda e: table.get(e.record.verdict.status))
    split = [*records, _reviewed_by(records[1], "deepseek/flash", status="REJECTED", hours=3,
                                    counts=(5, 5, 1, 4))]  # fmt: skip
    assert _standing(split)[0].status == status
    rival = [*records, _rival(records, status="REJECTED", hours=4, text="Fires on vowels.")]
    standing, _ = _standing(rival)
    assert standing.status == status
    if status == "VERIFIED":  # the earlier hypothesis keeps the standing, and its readings count
        assert standing.lead.record is records[1] and standing.lead.counts


def test_a_newer_needs_revision_hypothesis_leaves_a_verified_one_standing(records):  # noqa: F811
    """The L2/H3 case of #4780: a self-reviewed NEEDS_REVISION rival doesn't contest (#4900)."""
    records.append(_rival(records, status="NEEDS_REVISION", hours=4, text="Fires on nouns."))
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and unit.verified and standing.prior.record is records[1]
    roles = {e.record.submission.natural_language_explanation: e.role
             for e in unit.experiments if e.side == "positive"}  # fmt: skip
    assert roles["Fires on nouns."] == "failed_rival"


def test_needs_revision_does_not_join_rejected_against_verified(records):  # noqa: F811
    """VERIFIED, NEEDS_REVISION and REJECTED readings: one against one, so the scores decide.

    With no score on either side the pole is CONTESTED (#4900). Before, NEEDS_REVISION and
    REJECTED outvoted VERIFIED.
    """
    verified = records[1]
    records.append(
        _reviewed_by(
            verified, "deepseek/flash", status="NEEDS_REVISION", hours=3, counts=(5, 5, 1, 4)
        )
    )
    records.append(
        _reviewed_by(verified, "muse/spark", status="REJECTED", hours=5, counts=(2, 8, 0, 5))
    )
    standing, unit = _standing(records)
    assert standing.status == "CONTESTED" and not unit.verified


def test_independent_review_outweighs_self_review(records):  # noqa: F811
    verified = records[1]  # reviewed by anthropic/claude-test, proposed by openai/gpt-test
    records.append(
        _self_reviewed(_reviewed_by(verified, "x", status="REJECTED", hours=3, counts=(2, 8, 0, 5)))
    )
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and not standing.self_reviewed
    roles = {e.record.verdict.status: e.role for e in unit.experiments if e.side == "positive"}
    assert roles == {
        "NEEDS_REVISION": "earlier_hypothesis",
        "VERIFIED": "current",
        "REJECTED": "outweighed",
    }


def test_self_reviews_alone_can_contest(records):  # noqa: F811
    verified = _self_reviewed(records[1])
    records[1] = verified
    records.append(
        _reviewed_by(
            verified,
            verified.verdict.reviewer_agent_model,
            status="REJECTED",
            hours=3,
            counts=(2, 8, 0, 5),
        )
    )
    standing, _ = _standing(records)
    assert standing.status == "CONTESTED" and standing.self_reviewed


def test_remeasurement_replaces_its_reading(records):  # noqa: F811
    """Re-evaluating a probe set under a newer protocol replaces that reading.

    The old reading can't contest the new one.
    """
    verified = records[1]
    remeasured = verified.model_copy(
        update={
            "verdict": verified.verdict.model_copy(
                update={
                    "status": "NEEDS_REVISION",
                    "timestamp": verified.verdict.timestamp + timedelta(hours=3),
                }
            )
        }
    )
    records.append(remeasured)
    standing, unit = _standing(records)
    assert standing.status == "NEEDS_REVISION" and len(standing.readings) == 1
    assert [e.role for e in unit.experiments if e.record is verified] == ["remeasured"]


def test_revision_replaces_contested_hypothesis(records):  # noqa: F811
    verified = records[1]
    records.append(
        _reviewed_by(
            verified, "deepseek/flash", status="NEEDS_REVISION", hours=3, counts=(5, 5, 1, 4)
        )
    )
    revised = experiment(
        records[0], idx=117, status="VERIFIED", counts=(10, 0, 5, 0), hours=6, text="Narrower."
    )
    records.append(revised)
    standing, _ = _standing(records)
    assert standing.status == "VERIFIED"
    assert [e.record.submission.natural_language_explanation for e in standing.readings] == [
        "Narrower."
    ]


def test_a_second_failed_rival_leaves_it_contested(records):  # noqa: F811
    records.append(_rival(records, status="REJECTED", hours=4, text="Fires on vowels."))
    records.append(_rival(records, status="NEEDS_REVISION", hours=5, text="Fires on nouns."))
    standing, _ = _standing(records)
    assert standing.status == "CONTESTED" and standing.prior.record is records[1]


def test_a_verified_rival_settles_the_contest(records):  # noqa: F811
    """VERIFIED, then REJECTED, then a third hypothesis that is VERIFIED: VERIFIED."""
    records.append(_rival(records, status="REJECTED", hours=4, text="Fires on vowels."))
    records.append(
        _rival(records, status="VERIFIED", hours=6, text="Narrower.", counts=(10, 0, 5, 0))
    )
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and standing.prior is None and unit.verified
    assert standing.lead.record.submission.natural_language_explanation == "Narrower."


def test_rejected_then_verified_is_verified(records):  # noqa: F811
    """The ``records`` fixture: N117+ was NEEDS_REVISION, then a new hypothesis VERIFIED."""
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and standing.prior is None and unit.verified


def test_only_a_verified_earlier_hypothesis_contests(records):  # noqa: F811
    """N9+ was REJECTED; a new NEEDS_REVISION hypothesis is no contest."""
    records.append(
        experiment(
            records[0], idx=9, status="NEEDS_REVISION", counts=(5, 5, 1, 4), hours=3, text="Dates?"
        )
    )
    standing, _ = _standing(records, idx=9)
    assert standing.status == "NEEDS_REVISION" and standing.prior is None


def test_reviewer_agreement(records):  # noqa: F811
    from openinterpretability.builder.standing import reviewer_agreement
    from openinterpretability.builder.tree import build_tree

    verified = records[1]
    records.append(_reviewed_by(verified, "deepseek/flash", status="VERIFIED", hours=3))
    records.append(
        _reviewed_by(verified, "other/model", status="NEEDS_REVISION", hours=4, counts=(5, 5, 1, 4))
    )
    agreement = reviewer_agreement(build_tree(records, []))
    assert agreement["claims"] == 1 and agreement["unanimous"] == 0
    assert agreement["pairs"] == 3 and agreement["pair_agreement"] == round(1 / 3, 4)
    assert agreement["cross_model_pairs"] == 3
    assert agreement["cross_model_agreement"] == round(1 / 3, 4)


def _statuses(standing):
    return [e.record.verdict.status for e in standing.readings]


def _split(records, verified, rejected, hours=3):  # noqa: F811
    """N117+'s VERIFIED reading and a REJECTED one ``hours`` later, with these scores."""
    records[1] = _detection(records[1], verified)
    rejected_rec = _reviewed_by(records[1], "deepseek/flash", status="REJECTED", hours=hours,
                                counts=(2, 8, 0, 5))  # fmt: skip
    return _standing([*records, _detection(rejected_rec, rejected)])[0]


def test_an_even_split_puts_the_winning_side_first(records):  # noqa: F811
    verified = _split(list(records), 0.9, 0.6)
    assert verified.status == "VERIFIED" and _statuses(verified) == ["VERIFIED", "REJECTED"]
    rejected = _split(list(records), 0.6, 0.9)
    assert rejected.status == "REJECTED" and _statuses(rejected) == ["REJECTED", "VERIFIED"]
    older = _split(list(records), 0.6, 0.9, hours=-1)  # the REJECTED reading is older
    assert older.status == "REJECTED" and _statuses(older) == ["REJECTED", "VERIFIED"]
    contested = _split(list(records), 0.8, 0.8)  # newest first
    assert contested.status == "CONTESTED" and _statuses(contested) == ["REJECTED", "VERIFIED"]


def test_scores_exactly_the_margin_apart_settle_a_split(records):  # noqa: F811
    assert abs(0.11 - 0.06) == st.SCORE_MARGIN
    assert _split(list(records), 0.11, 0.06).status == "VERIFIED"
    assert _split(list(records), 0.06, 0.11).status == "REJECTED"


@pytest.mark.parametrize("missing", ["VERIFIED", "REJECTED"])
def test_a_split_with_one_score_missing_is_contested(records, monkeypatch, missing):  # noqa: F811
    table = {"VERIFIED": 0.9, "REJECTED": 0.1, missing: None}
    monkeypatch.setattr(st, "score", lambda e: table[e.record.verdict.status])
    assert _split(list(records), 0.9, 0.1).status == "CONTESTED"


def test_score_falls_back_to_the_probe_balanced_accuracy(records):  # noqa: F811
    _, unit = _standing(records)
    (e,) = [e for e in unit.experiments if e.record is records[1]]
    assert score(e) is None  # no held-out score and no probe metrics
    metrics = ReviewMetrics(specificity=1.0, sensitivity=0.5, balanced_accuracy=0.75,
                            counterfactual_reproduction=None, lookahead_consistent=True,
                            independent_review=True, gates={})  # fmt: skip
    records[1] = records[1].model_copy(
        update={"verdict": records[1].verdict.model_copy(update={"metrics": metrics})}
    )
    _, unit = _standing(records)
    (e,) = [e for e in unit.experiments if e.record is records[1]]
    assert score(e) == 0.75
    records[1] = _detection(records[1], 0.6)
    _, unit = _standing(records)
    (e,) = [e for e in unit.experiments if e.record is records[1]]
    assert score(e) == 0.6


def _roles(unit):
    return {
        e.record.submission.natural_language_explanation: e.role
        for e in unit.experiments
        if e.side == "positive"
    }


def test_a_contested_hypothesis_is_listed_with_its_roles(records):  # noqa: F811
    records.append(_rival(records, status="REJECTED", hours=4, text="Fires on vowels."))
    (model,) = build_tree(records, [])
    (unit,) = [n for layer in model.layers for n in layer.units if n.idx == 117]
    standing = unit.standings["positive"]
    assert standing.status == "CONTESTED"
    assert _roles(unit)[records[1].submission.natural_language_explanation] == (
        "contested_hypothesis"
    )
    assert _roles(unit)["Fires on vowels."] == "current"
    assert all(e.standing is standing for e in unit.experiments if e.side == "positive")
    (row,) = contested_units([model])
    assert row == {"model": unit.model, "unit": unit, "side": "positive", "standing": standing}


def test_a_verified_hypothesis_that_outscores_its_rival_keeps_the_standing(
    records,  # noqa: F811
    monkeypatch,
):
    table = {"VERIFIED": 0.9, "REJECTED": 0.7}
    monkeypatch.setattr(st, "score", lambda e: table.get(e.record.verdict.status))
    records.append(_rival(records, status="REJECTED", hours=4, text="Fires on vowels."))
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and standing.prior.record is records[1]
    assert [e.record for e in standing.readings] == [records[1]]
    roles = _roles(unit)
    assert roles[records[1].submission.natural_language_explanation] == "standing_hypothesis"
    assert roles["Fires on vowels."] == "failed_rival"


def test_the_prior_is_the_newest_earlier_verified_hypothesis(records):  # noqa: F811
    records.append(_rival(records, status="VERIFIED", hours=3, text="Second.",
                          counts=(10, 0, 5, 0)))  # fmt: skip
    records.append(_rival(records, status="REJECTED", hours=4, text="Fires on vowels."))
    standing, _ = _standing(records)
    assert standing.status == "CONTESTED"
    assert standing.prior.record.submission.natural_language_explanation == "Second."


def test_an_experiment_set_aside_by_an_erratum_does_not_count(record):
    newer = later(record, 1, "REJECTED")
    (model,) = build_tree([record, newer], [erratum(records=[record_ref(newer)])])
    (unit,) = model.layers[0].units
    standing = unit.standings["positive"]
    assert standing.status == "VERIFIED" and [e.record for e in standing.readings] == [record]
    assert {e.record.verdict.status: e.role for e in unit.experiments} == {
        "VERIFIED": "current", "REJECTED": "set_aside",
    }  # fmt: skip
    assert all(e.standing is standing for e in unit.experiments)


def test_when_every_experiment_is_set_aside_none_is_left_out(record):
    newer = later(record, 1, "REJECTED")
    rule = erratum(rule={"models": [record.submission.identifier.model_name]})
    (model,) = build_tree([record, newer], [rule])
    (unit,) = model.layers[0].units
    assert unit.standings["positive"].status == "REJECTED"
    # One probe set, so the newer experiment re-measures the older one. Neither is set aside.
    assert sorted(e.role for e in unit.experiments) == ["current", "remeasured"]


def test_no_repeated_claim_gives_an_empty_agreement(records):  # noqa: F811
    assert reviewer_agreement(build_tree(records, [])) == {
        "claims": 0, "unanimous": 0, "pairs": 0, "pair_agreement": None,
        "cross_model_pairs": 0, "cross_model_agreement": None,
    }  # fmt: skip


def test_agreement_adds_up_over_claims(records):  # noqa: F811
    for base, status, counts in ((records[1], "VERIFIED", (10, 0, 5, 0)),
                                 (records[3], "REJECTED", (3, 3, 1, 3))):  # fmt: skip
        for reviewer, hours in (("deepseek/flash", 3), ("muse/spark", 4)):
            records.append(_reviewed_by(base, reviewer, status=status, hours=hours, counts=counts))
    assert reviewer_agreement(build_tree(records, [])) == {
        "claims": 2, "unanimous": 2, "pairs": 6, "pair_agreement": 1.0,
        "cross_model_pairs": 6, "cross_model_agreement": 1.0,
    }  # fmt: skip
