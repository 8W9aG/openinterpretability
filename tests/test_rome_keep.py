"""The helpers of scripts/rome_keep.py and scripts/rome_keep_stats.py (#5167)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import rome_keep as rk  # noqa: E402
import rome_keep_stats as stats  # noqa: E402


def _problem(seed=0, d=4, hidden=6):
    g = torch.Generator().manual_seed(seed)
    weight, bias = torch.randn(d, hidden, generator=g), torch.randn(d, generator=g)
    keys = torch.randn(20, hidden, generator=g)
    cov = keys.T @ keys / 20 + 0.1 * torch.eye(hidden)
    return weight, bias, cov, torch.randn(hidden, generator=g), torch.randn(d, generator=g)


def test_the_rank_one_update_stores_the_new_value_at_the_key():
    weight, bias, cov, key, value = _problem()
    lam, u = rk.rome_update(weight, bias, cov, key, value)
    assert torch.allclose((weight + torch.outer(lam, u)) @ key + bias, value, atol=1e-4)
    assert torch.allclose(u, torch.linalg.solve(cov, key))


def test_the_update_changes_each_column_by_u_j_times_lambda():
    weight, bias, cov, key, value = _problem(1)
    lam, u = rk.rome_update(weight, bias, cov, key, value)
    change = torch.outer(lam, u)
    for j in range(len(u)):  # neuron j's write moves along Λ, by u_j
        assert torch.allclose(change[:, j], u[j] * lam)


def test_the_random_null_has_the_same_norms_and_follows_its_seed():
    lam, u = torch.randn(4), torch.randn(6)
    a, b = rk.random_rank_one(lam, u, 3)
    assert float(a.norm()) == pytest.approx(float(lam.norm()))
    assert float(b.norm()) == pytest.approx(float(u.norm()))
    assert float(torch.outer(a, b).norm()) == pytest.approx(float(torch.outer(lam, u).norm()))
    again, other = rk.random_rank_one(lam, u, 3), rk.random_rank_one(lam, u, 4)
    assert torch.equal(a, again[0]) and not torch.equal(a, other[0])


def test_alignment_is_the_absolute_cosine_after_the_gain():
    w_in, gain = torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0])
    assert rk.alignment(w_in, gain, torch.tensor([-1.0, 0.0])) == pytest.approx(1.0)
    assert rk.alignment(w_in, gain, torch.tensor([0.0, 3.0])) == pytest.approx(0.0)
    assert rk.alignment(w_in, gain, torch.tensor([1.0, 2.0])) == pytest.approx(2 / 8**0.5)


def test_u_loading_ranks_the_verified_neurons_among_all():
    u = torch.tensor([0.1, -5.0, 0.2, 3.0, 0.0])
    got = rk.u_loading(u, [1, 0])
    assert got["percentiles"] == [1.0, 0.25] and got["median_percentile"] == 0.625
    assert rk.u_loading(u, [])["median_percentile"] is None
    assert rk.param_name(2) == "gpt_neox.layers.2.mlp.dense_4h_to_h.weight"


def _pole(slug, kind, fact, layer, keeps, alignment=0.0, pearson=0.99, ratio=1.0):
    return {"table": "pole", "condition": kind, "fact": fact, "edit_layer": 2, "layer": layer,
            "slug": slug, "base_passes": True, "keeps": keeps, "alignment": alignment,
            "pearson": pearson, "ratio": ratio}  # fmt: skip


ROWS = [
    _pole("a", "rome", 0, 3, False, 0.9), _pole("b", "rome", 0, 3, True, 0.1),
    _pole("c", "rome", 0, 4, True, 0.2), _pole("low", "rome", 0, 1, True, pearson=1.0),
    _pole("a", "random", 0, 3, False, 0.1), _pole("b", "random", 0, 3, False, 0.5),
    _pole("c", "random", 0, 4, True, 0.3), _pole("low", "random", 0, 2, True, pearson=1.0),
    {"table": "base", "slug": "a", "readings": {"specificity": 0.8, "sensitivity": 0.9}},
    {"table": "base", "slug": "b", "readings": {"specificity": 0.9, "sensitivity": 0.9}},
    {"table": "base", "slug": "c", "readings": {"specificity": 1.0, "sensitivity": 1.0}},
    {"table": "summary", "condition": "rome", "u_loading": {"median_percentile": 0.7}},
    {"table": "summary", "condition": "random", "u_loading": {"median_percentile": 0.4}},
]  # fmt: skip


def test_the_check_counts_changed_readings_at_or_below_the_edit():
    assert stats.check(ROWS) == {"table": "check", "readings": 2, "changed": 0}
    moved = [*ROWS, _pole("x", "rome", 1, 2, True, pearson=1.0, ratio=0.9)]
    assert stats.check(moved)["changed"] == 1


def test_flips_count_by_kind_and_fact():
    got = stats.flips(ROWS)
    assert got["rome"]["total"] == 1 and got["random"]["total"] == 2
    assert got["facts_rome_fewer"] == 1 and got["random"]["distinct_poles"] == 2


def test_alignment_auroc_and_its_shuffle_null():
    assert stats.auroc([0.9], [0.1, 0.2]) == 1.0 and stats.auroc([], [0.1]) is None
    got = stats.alignment_test(ROWS, "rome", shuffles=200)
    assert got["auroc"] == 1.0 and 0 < got["p"] <= 1


def test_margins_and_loadings():
    got = stats.margins(ROWS)
    assert sum(got["quarter_sizes"]) == 3 and sum(got["random"]) == 2
    loads = stats.u_loadings(ROWS)
    assert loads["rome"]["above_half"] == 1 and loads["random"]["mean"] == 0.4
