"""The weight-only kurtosis signal (scripts/weight_kurtosis.py, issue #3226)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import weight_kurtosis as wk  # noqa: E402


def _reference(row: torch.Tensor, unembed: torch.Tensor) -> tuple[float, float]:
    """The kurtosis and skewness of one row, as ``jlens._decode`` standardises scores."""
    scores = unembed @ row
    z = (scores - scores.mean()) / scores.std()
    return float((z**4).mean()), float((z**3).mean())


def test_moments_match_the_lens_in_chunks(monkeypatch):
    monkeypatch.setattr(wk, "CHUNK", 3)  # several chunks, the last one short
    gen = torch.Generator().manual_seed(0)
    unembed, rows = torch.randn(400, 8, generator=gen), torch.randn(7, 8, generator=gen)
    kurt, skew = wk.moments(rows, unembed)
    for i in range(len(rows)):
        k, s = _reference(rows[i], unembed)
        assert float(kurt[i]) == pytest.approx(k, rel=1e-5)
        assert float(skew[i]) == pytest.approx(s, rel=1e-4, abs=1e-6)


def test_kurtosis_is_the_same_for_both_signs_and_skew_flips():
    gen = torch.Generator().manual_seed(1)
    unembed, rows = torch.randn(300, 6, generator=gen), torch.randn(4, 6, generator=gen)
    k_pos, s_pos = wk.moments(rows, unembed)
    k_neg, s_neg = wk.moments(-rows, unembed)
    assert torch.allclose(k_pos, k_neg)
    assert torch.allclose(s_pos, -s_neg)


def test_a_one_token_direction_scores_far_above_a_gaussian():
    unembed = torch.eye(500, 500)[:, :50] + 0.0  # 500 tokens, d = 50: tokens 0–49 are axes
    spike = torch.zeros(1, 50)
    spike[0, 0] = 1.0  # reads one token only
    spread = torch.ones(1, 50)  # reads 50 tokens alike
    assert float(wk.moments(spike, unembed)[0]) > float(wk.moments(spread, unembed)[0]) > 3


def test_percentile_counts_ties_as_half():
    values = [1.0, 2.0, 2.0, 3.0]
    assert wk.percentile(values, 0.5) == 0.0
    assert wk.percentile(values, 2.0) == pytest.approx(0.5)
    assert wk.percentile(values, 4.0) == 1.0


def test_add_kurtoses_signs_the_skew_by_pole():
    stats = {
        "kurtosis_out": torch.tensor([3.0, 5.0, 9.0], dtype=torch.float64),
        "kurtosis_in": torch.tensor([4.0, 3.0, 3.5], dtype=torch.float64),
        "skew_out": torch.tensor([0.0, 0.5, -1.0], dtype=torch.float64),
    }
    units = [
        {"model_name": "m", "layer": 0, "neuron_idx": 2, "pole": "positive"},
        {"model_name": "m", "layer": 0, "neuron_idx": 2, "pole": "negative"},
    ]
    wk.add_kurtoses(units, {"m": {0: stats}})
    assert units[0]["kurtosis_out"] == units[1]["kurtosis_out"] == 9.0
    assert units[0]["kurtosis_out_pct"] == pytest.approx(5 / 6)  # top of 3, ties as half
    assert units[0]["skew_out_signed"] == -1.0 and units[1]["skew_out_signed"] == 1.0
