"""Registry coverage: mean-ablating the poles that aren't VERIFIED (openinterp/coverage.py)."""

import pytest
import torch

from openinterp.coverage import keep_poles, layer_row, mean_loss, random_poles, recovered, windows

TEXTS = ["abcdefghijklmnopqrstuvwxyz" * 2 + str(i) for i in range(12)]


def test_keep_poles_keeps_only_the_named_parts():
    means = (torch.tensor([0.5, 0.5]), torch.tensor([-0.5, -0.5]))
    act = torch.tensor([[2.0, -3.0], [-1.0, 4.0]])
    out = keep_poles(means, {(0, "positive"), (1, "negative")}, 2)(act)
    # Neuron 0 keeps its positive part, neuron 1 its negative part. The rest is the mean.
    assert out.tolist() == [[2.0 - 0.5, 0.5 - 3.0], [0.0 - 0.5, 0.5 + 0.0]]
    assert keep_poles(means, set(), 2)(act).tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_recovered_share():
    assert recovered(clean=2.0, ablated=3.0, kept=2.25) == pytest.approx(0.75)
    assert recovered(clean=2.0, ablated=2.0, kept=2.0) is None  # nothing to recover


def test_random_poles_match_the_count_and_avoid_the_verified():
    verified = {(0, "positive"), (3, "negative")}
    drawn = random_poles(verified, 5, seed=1)
    assert len(drawn) == 2 and not drawn & verified
    assert drawn == random_poles(verified, 5, seed=1)
    everything = {(i, p) for i in range(2) for p in ("positive", "negative")}
    assert random_poles(everything, 2, seed=1) == set()  # no other poles to draw


def test_windows_are_disjoint_and_need_enough_text(toy_runner):
    fit, test = windows(toy_runner, TEXTS, 4, 16, seed=0)
    assert fit.shape == test.shape == (4, 16)
    with pytest.raises(ValueError):
        windows(toy_runner, TEXTS, 7, 16, seed=0)


def test_keeping_every_pole_recovers_everything(toy_runner):
    fit, test = windows(toy_runner, TEXTS, 4, 16, seed=0)
    width = toy_runner.mlp_width(1)
    everything = {(i, p) for i in range(width) for p in ("positive", "negative")}
    clean = mean_loss(toy_runner, test, batch=2)
    row = layer_row(toy_runner, fit, test, 1, everything, clean, batch=2, seed=0)
    assert row["loss_kept_verified"] == pytest.approx(clean, abs=1e-5)
    if row["loss_ablated"] > clean:
        assert row["recovered"] == pytest.approx(1.0, abs=1e-4)


def test_a_layer_without_verified_poles_has_no_share(toy_runner):
    fit, test = windows(toy_runner, TEXTS, 4, 16, seed=0)
    clean = mean_loss(toy_runner, test, batch=2)
    row = layer_row(toy_runner, fit, test, 0, set(), clean, batch=2, seed=0)
    assert "recovered" not in row and row["verified_poles"] == 0
