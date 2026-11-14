"""The perturbations and metrics behind docs/analyses/score-sanity.md."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from score_sanity import balanced, beats_chance, check, f1, perturb, precision, recall  # noqa: E402

# (answer, label): 3 of 4 active windows found, 1 of 4 inactive ones called active.
PAIRS = [(True, True)] * 3 + [(False, True)] + [(True, False)] + [(False, False)] * 3


def test_metrics():
    assert balanced(PAIRS) == 0.75 and recall(PAIRS) == 0.75 and precision(PAIRS) == 0.75
    assert f1(PAIRS) == 0.75
    assert balanced([(True, True)]) is None  # one class only


def test_perturb_flips_only_the_named_answers_and_never_the_labels():
    rng = random.Random(0)
    missing = perturb(PAIRS, "missing", 1.0, rng)
    assert [y for _, y in missing] == [y for _, y in PAIRS]
    assert not any(a for a, _ in missing)
    extra = perturb(PAIRS, "extra", 1.0, rng)
    assert all(a for a, _ in extra)
    assert perturb(PAIRS, "extra", 0.0, rng) == PAIRS


def test_beats_chance():
    assert beats_chance(PAIRS)
    assert not beats_chance([(True, True), (True, False)])  # calls everything active


def test_balanced_accuracy_drops_under_both_tests_and_recall_not_under_extra():
    result = check([PAIRS])
    for test in ("missing", "extra"):
        assert result[test]["0.5"]["balanced_accuracy"]["share_dropping"] == 1.0
    assert result["extra"]["0.5"]["recall"]["mean_change"] > 0
