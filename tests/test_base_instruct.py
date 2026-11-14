"""Helpers of the base ↔ instruct comparison (scripts/base_instruct.py, issue #3375)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import base_instruct as bi  # noqa: E402


def test_pole_max_reads_each_pole_as_a_magnitude():
    acts = torch.tensor([-3.0, 1.0, 2.0])
    assert bi.pole_max(acts, "positive") == 2.0
    assert bi.pole_max(acts, "negative") == 3.0


def test_pearson_and_a_constant_vector():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert bi.pearson(a, 2 * a + 1) == pytest.approx(1.0)
    assert bi.pearson(a, -a) == pytest.approx(-1.0)
    assert bi.pearson(a, torch.ones(4)) is None


def test_wilson_interval_brackets_the_share():
    lo, hi = bi.wilson(30, 40)
    assert 0 <= lo < 0.75 < hi <= 1
    assert bi.wilson(0, 0) == [0.0, 1.0]
