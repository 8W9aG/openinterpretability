"""A pole's standing and the reviewer agreement (builder/standing.py)."""

from openinterpretability.builder.standing import reviewer_agreement
from openinterpretability.builder.tree import build_tree
from openinterpretability.core.explanation import ExplanationScores, JudgeScore
from tests.test_compiler import _reviewed_by, _standing, records  # noqa: F401  (a fixture)


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
