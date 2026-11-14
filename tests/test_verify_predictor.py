"""The verify predictor's features and scores (scripts/verify_predictor.py, issue #3372)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import verify_predictor as vp  # noqa: E402


def test_the_active_share_counts_bins_above_a_quarter_of_the_top():
    hist = {"bin_edges": [0.0, 1.0, 2.0, 3.0, 4.0], "counts": [6, 2, 1, 1]}
    got = vp.histogram_features(hist)
    assert got["active_share"] == pytest.approx(0.4)  # the bins from 1.0 = 0.25 of 4.0
    assert got["floor_share"] == pytest.approx(0.6)
    assert vp.histogram_features(None) == {"active_share": None, "floor_share": None}


def test_auc_and_the_gain_over_the_base_rate():
    y = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    perfect = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.1], dtype=torch.float64)
    assert vp.auc(perfect, y) == 1.0
    assert vp.auc(torch.full_like(y, 0.3), y) == 0.5
    gain = vp.gain_interval(perfect, 0.4, y)
    assert gain["brier_gain"][0] > 0 and gain["log_gain"][0] > 0
    assert gain["brier_gain"][1] <= gain["brier_gain"][0] <= gain["brier_gain"][2]


def test_the_fit_recovers_a_clear_signal():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(400, 1, generator=g, dtype=torch.float64)
    y = (torch.rand(400, generator=g, dtype=torch.float64) < torch.sigmoid(3 * x[:, 0])).double()
    w = vp.fit(x, y, penalty=0.01)
    assert 2.0 < float(w[1]) < 4.0
