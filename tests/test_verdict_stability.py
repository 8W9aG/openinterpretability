"""Same-reviewer agreement and the variance split (scripts/verdict_stability.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "verdict_stability", _SCRIPTS / "verdict_stability.py"
)
vs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vs)


def test_pairwise_and_fleiss():
    assert vs.pairwise([["a", "a", "a"], ["a", "b", "b"]]) == pytest.approx(4 / 6)
    assert vs.fleiss([["a", "a", "a"], ["b", "b", "b"]]) == pytest.approx(1.0)
    assert vs.fleiss([["a", "b"], ["b", "a"]]) == pytest.approx(-1.0)


def _arm(ba, outcome="VERIFIED", spec=True):
    return {"balanced_accuracy": ba, "outcome": outcome, "gates": {"specificity": spec, "x": True}}


def test_noisy_repeats_show_as_same_reviewer_noise():
    # Three submissions at 0.6, 0.8, 1.0. The Sonnet repeats scatter ±0.05; the other
    # reviewers agree with Sonnet's first review, so no reviewer or interaction effect.
    rows = []
    for base in (0.6, 0.8, 1.0):
        row = {
            "sonnet": _arm(base),
            "sonnet2": _arm(base + 0.05, spec=False),
            "sonnet3": _arm(base - 0.05),
        }
        row |= {"opus": _arm(base), "haiku": _arm(base)}
        rows.append(row)
    v = vs.variance(rows)
    assert v["same_reviewer_noise"] == pytest.approx(0.0025)
    assert v["reviewer"] == pytest.approx(0, abs=1e-12) and v["interaction"] == pytest.approx(
        0, abs=1e-12
    )
    assert v["submission"] == pytest.approx(0.04)
    a = vs.agreement(rows)
    assert a["verified_pairwise"] == 1.0 and a["gate_flip_rate"]["specificity"] == pytest.approx(
        2 / 3
    )
    assert a["gate_flip_rate"]["x"] == 0
