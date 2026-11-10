"""Tests for scripts/memory_benchmark.py: the comparison with the baseline."""

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "memory_benchmark", Path(__file__).parents[1] / "scripts" / "memory_benchmark.py"
)
bench = importlib.util.module_from_spec(_SPEC)
sys.modules["memory_benchmark"] = bench  # a dataclass looks its module up there
_SPEC.loader.exec_module(bench)

BASE = {"m": {"scan_mb": 1000, "review_mb": 2000}}


def test_a_peak_within_the_tolerance_passes():
    now = {"m": {"scan_mb": 1200, "review_mb": 1700}}
    assert bench.compare(BASE, now, 0.25) == ([], [])


def test_a_peak_above_the_tolerance_fails_and_one_below_is_reported():
    now = {"m": {"scan_mb": 1300, "review_mb": 1400}}
    above, below = bench.compare(BASE, now, 0.25)
    assert above == ["m scan_mb: 1300 MB, baseline 1000 MB"]
    assert below == ["m review_mb: 1400 MB, baseline 2000 MB"]


def test_a_model_missing_from_the_baseline_is_not_compared():
    assert bench.compare(BASE, {"new": {"scan_mb": 9999, "review_mb": 9999}}, 0.25) == ([], [])


def test_the_committed_baseline_covers_every_benchmark():
    baseline = json.loads(bench.BASELINE.read_text())
    assert set(baseline["models"]) == {b.model for b in bench.BENCHMARKS}
    assert baseline["max_tokens"] == bench.MAX_TOKENS
    for b in bench.BENCHMARKS:  # each record to reproduce is in the registry
        assert (bench.ROOT / "wiki" / "experiments" / b.record).exists()
