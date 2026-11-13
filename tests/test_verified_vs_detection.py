"""The VERIFIED against held-out detection comparison (scripts/verified_vs_detection.py)."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "verified_vs_detection", Path(__file__).parents[1] / "scripts" / "verified_vs_detection.py"
)
vvd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vvd)


def _row(status, judge, detection, slug="a", time="1", fuzzing=None):
    return {"status": status, "judge": judge, "detection": detection, "fuzzing": fuzzing,
            "slug": slug, "time": time}  # fmt: skip


def test_judge_names_are_normalised():
    assert vvd.normalise("external/deepseek-flash[1m]") == "deepseek-flash"
    assert vvd.normalise("external/muse-spark-1-3-contributor") == "muse-spark-1.3"
    assert vvd.normalise(None) == "unknown"


def test_within_judge_removes_a_judge_level_difference():
    # Judge A scores everything 0.2 higher and judges mostly VERIFIED records. Within
    # each judge, VERIFIED is 0.05 higher. The raw difference is inflated, and the
    # within-judge difference is not.
    data = [_row("VERIFIED", "A", 0.85)] * 30 + [_row("REJECTED", "A", 0.80)] * 10
    data += [_row("VERIFIED", "B", 0.65)] * 10 + [_row("REJECTED", "B", 0.60)] * 30
    assert vvd.within_judge(data, "detection") == pytest.approx(0.05)
    ok, other = vvd.arms(data, "detection")
    assert vvd.mean(ok) - vvd.mean(other) > 0.1


def test_small_judges_are_left_out_and_missing_scores_skipped():
    data = [_row("VERIFIED", "A", 0.7)] * 10 + [_row("REJECTED", "A", 0.6)] * 10
    data += [_row("VERIFIED", "tiny", 0.0)] * 3 + [_row("REJECTED", "tiny", 1.0)] * 3
    assert vvd.within_judge(data, "detection") == pytest.approx(0.1)
    assert vvd.within_judge(data, "fuzzing") is None  # no fuzzing scores at all


def test_latest_record_per_unit():
    data = [
        _row("REJECTED", "A", 0.5, slug="u", time="1"),
        _row("VERIFIED", "A", 0.9, slug="u", time="2"),
    ]
    assert vvd.latest_per_unit(data) == [data[1]]


def test_bootstrap_interval_contains_the_estimate():
    data = [_row("VERIFIED", "A", 0.7 + i / 100) for i in range(20)]
    data += [_row("REJECTED", "A", 0.6 + i / 100) for i in range(20)]
    low, high = vvd.bootstrap(data, lambda d: vvd.within_judge(d, "detection"))
    assert low <= 0.1 <= high
