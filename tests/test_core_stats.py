"""Rank correlation (core/stats.py)."""

import pytest
import torch

from openinterpretability.core.stats import spearman


def test_spearman_is_zero_with_fewer_than_three_points():
    assert spearman(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])) == 0.0
    assert spearman(torch.tensor([]), torch.tensor([])) == 0.0


def test_spearman_of_a_monotonic_pair_is_one():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x.pow(3)) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
