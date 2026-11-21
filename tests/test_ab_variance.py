"""The noise measures of a null prompt A/B test (scripts/ab_variance.py, #3198)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("ab_variance", _SCRIPTS / "ab_variance.py")
ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ab)


def _row(sub, review, ba, verified=False, spec=0.9):
    return {"submission": sub, "review": review, "balanced_accuracy": ba, "verified": verified,
            "specificity_margin": spec - ab.MIN_SPECIFICITY, "sensitivity_margin": 0.2,
            "probes": 20}  # fmt: skip


def test_units_needed_follows_the_power_formula():
    # n = ((1.96 + 0.84) · sd / shift)², rounded up.
    assert ab.units_needed(0.14, 0.05) == 62
    assert ab.units_needed(0.10, 0.05) == 32
    assert ab.units_needed(0.0, 0.05) == 0


def test_row_keeps_the_margins_to_the_gates():
    r = ab.row("c01", "opus", {"balanced_accuracy": 0.7, "outcome": "REJECTED",
                               "specificity": 0.75, "sensitivity": 0.5})  # fmt: skip
    assert not r["verified"] and r["probes"] is None
    assert r["specificity_margin"] == pytest.approx(-0.05)
    assert r["sensitivity_margin"] == pytest.approx(-0.10)


def test_paired_sd_and_within_variance():
    by_review = {
        "a": {"s1": _row("s1", "a", 0.5), "s2": _row("s2", "a", 0.8), "s3": _row("s3", "a", 0.6)},
        "b": {"s1": _row("s1", "b", 0.6), "s2": _row("s2", "b", 0.8), "s3": _row("s3", "b", 0.8)},
    }
    # Differences 0.1, 0.0, 0.2: SD 0.1.
    assert ab.paired_sd(by_review, "a", "b") == pytest.approx(0.1)
    # Pairs (0.5, 0.6), (0.8, 0.8), (0.6, 0.8): variances 0.005, 0, 0.02.
    assert ab.within_variance(by_review, ("a", "b")) == pytest.approx(0.025 / 3)


def test_flip_rate_and_near_gate():
    by_review = {
        "a": {"s1": _row("s1", "a", 0.9, True, 0.82), "s2": _row("s2", "a", 0.9, True, 0.95)},
        "b": {"s1": _row("s1", "b", 0.9, False, 0.78), "s2": _row("s2", "b", 0.9, True, 0.95)},
    }
    assert ab.flip_rate(by_review, ("a", "b")) == pytest.approx(0.5)
    assert ab.near_gate(by_review, ("a", "b")) == pytest.approx(0.5)  # 0.82 and 0.78


def test_cuped_removes_the_covariates_share():
    out = ab.cuped([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert out["removed"] == pytest.approx(out["rho"] ** 2)
    assert out["after"] == pytest.approx(out["before"] * (1 - out["rho"] ** 2))


def test_summary_on_replicates_that_differ_only_by_chance():
    reviews = (*ab.TWENTY, *ab.THIRTY, "opus")
    rows = [_row(f"s{i}", r, 0.5 + 0.1 * i + 0.01 * j * (-1) ** i)
            for i in range(5) for j, r in enumerate(reviews)]  # fmt: skip
    out = ab.summary(rows)
    assert out["submissions"] == 5
    assert out["replicates"]["2"]["sd"] == pytest.approx(out["replicates"]["1"]["sd"] / 2**0.5)
    assert out["units_ba_20"] == ab.units_needed(out["paired_sd_20"], ab.SHIFT_BA)
    assert 0 <= out["cuped_opus"]["removed"] <= 1
