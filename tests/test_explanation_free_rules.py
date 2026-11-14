"""Explanation-free rules on scoring windows (scripts/explanation_free_rules.py)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "explanation_free_rules", Path(__file__).parents[1] / "scripts" / "explanation_free_rules.py"
)
rules = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rules)


def test_balanced_accuracy_and_contiguity():
    assert rules.balanced([(True, True), (False, False), (True, False), (True, True)]) == 0.75
    assert rules.balanced([(True, True)]) is None
    assert rules.contiguous([3, 4, 5]) and not rules.contiguous([3, 5]) and not rules.contiguous([])


def win(marks, label):
    return SimpleNamespace(highlighted=marks, label=label, judged=None, tokens=[])


def test_a_leaky_item_set_is_caught():
    # Correct items mark a run, decoys scatter their marks: the contiguity rule is perfect.
    fuz = [win([2, 3, 4], True), win([5, 6], True), win([1, 7, 9], False), win([0, 4], False)]
    det = [SimpleNamespace(tokens=["a"] * 32, label=True, judged=None)]
    det += [SimpleNamespace(tokens=["a"] * 20, label=False, judged=None)]
    out = rules.pooled([SimpleNamespace(detection_windows=det, fuzzing_windows=fuz)])
    assert out["fuzzing_marks_contiguous"] == 1.0
    assert out["detection_length_rule"] == 1.0
