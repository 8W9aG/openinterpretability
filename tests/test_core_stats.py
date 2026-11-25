"""Probe statistics and rank correlation (core/stats.py)."""

import pytest
import torch

from openinterpretability.core.stats import (
    EffectiveCounts,
    balanced_accuracy,
    cluster_texts,
    effective_counts,
    jaccard,
    p_balanced_above,
    rank_columns,
    spearman,
    spearman_columns,
)


def test_spearman_is_zero_with_fewer_than_three_points():
    assert spearman(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])) == 0.0
    assert spearman(torch.tensor([]), torch.tensor([])) == 0.0


def test_spearman_of_a_monotonic_pair_is_one():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x.pow(3)) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_measures_three_points():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert spearman(x, x) == pytest.approx(1.0)


def test_jaccard_is_the_shared_share_of_all_words():
    assert jaccard(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)
    assert jaccard(frozenset("abc"), frozenset("abd")) == pytest.approx(0.5)


def test_jaccard_of_two_empty_sets_is_one():
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_cluster_texts_joins_an_overlap_at_the_threshold():
    # "a b c" and "a b d" share 2 of 4 words: exactly the threshold.
    assert cluster_texts(["a b c", "a b d", "x y z"], 0.5) == [0, 0, 1]
    assert cluster_texts(["a b c", "a b d", "x y z"], 0.51) == [0, 1, 2]


def test_cluster_texts_joins_a_chain_into_one_cluster():
    # Neighbours share 2 of 4 words, but the two ends share none.
    texts = ["x y a b", "a b c d", "c d e f", "e f g h", "q r s t"]
    assert jaccard(frozenset("xyab"), frozenset("efgh")) == 0.0
    assert cluster_texts(texts, 0.3) == [0, 0, 0, 0, 1]
    assert cluster_texts(list(reversed(texts)), 0.3) == [0, 1, 1, 1, 1]


def test_cluster_texts_of_no_texts_is_empty():
    assert cluster_texts([], 0.5) == []


def test_effective_counts_scores_each_cluster_by_its_mean_pass_rate():
    counts = effective_counts(["a b c", "a b d", "x y z"], [True, False, True], 0.5)
    assert counts == EffectiveCounts(raw=3, clusters=2, passed=1.5)
    assert counts.rate == pytest.approx(0.75)


def test_effective_counts_refuses_texts_and_results_of_different_lengths():
    with pytest.raises(ValueError):
        effective_counts(["a", "b"], [True], 0.5)
    with pytest.raises(ValueError):
        effective_counts(["a"], [True, False], 0.5)


def test_balanced_accuracy_with_no_rates_is_zero():
    none = EffectiveCounts(raw=0, clusters=0, passed=0.0)
    assert balanced_accuracy(none, none) == 0.0


def test_balanced_accuracy_is_the_mean_of_the_rates_that_exist():
    none = EffectiveCounts(raw=0, clusters=0, passed=0.0)
    half = EffectiveCounts(raw=4, clusters=4, passed=2.0)
    full = EffectiveCounts(raw=2, clusters=2, passed=2.0)
    assert balanced_accuracy(half, full) == pytest.approx(0.75)
    assert balanced_accuracy(none, full) == pytest.approx(1.0)


def test_p_balanced_above_defaults_are_the_recorded_ones():
    # Stored records hold these values. Other priors, a changed sample count or seed,
    # or no seed at all would score the same record differently.
    spec = EffectiveCounts(raw=5, clusters=5, passed=3.0)
    sens = EffectiveCounts(raw=6, clusters=6, passed=4.0)
    assert p_balanced_above(spec, sens, 0.6) == 0.5115
    assert p_balanced_above(spec, sens, 0.6) == p_balanced_above(
        spec, sens, 0.6, samples=4000, seed=0
    )
    swapped = (EffectiveCounts(4, 4, 1.5), EffectiveCounts(3, 3, 3.0))
    assert p_balanced_above(*swapped, 0.5) == 0.80275


def test_rank_columns_gives_tied_values_their_mean_rank():
    x = torch.tensor([[3.0], [1.0], [2.0], [2.0], [0.0], [0.0], [2.0]])
    ranks = rank_columns(x)[:, 0].tolist()
    assert ranks == [6.0, 2.0, 4.0, 4.0, 0.5, 0.5, 4.0]


def test_rank_columns_of_integers_keeps_fractional_ranks():
    ranks = rank_columns(torch.tensor([[5], [7], [5]]))
    assert ranks.dtype == torch.float32
    assert ranks[:, 0].tolist() == [0.5, 2.0, 0.5]


def test_rank_columns_is_the_same_for_every_chunk_size():
    x = torch.randint(0, 4, (9, 7), generator=torch.Generator().manual_seed(0)).float()
    whole = rank_columns(x)
    for chunk in (1, 2, 3, 7):
        assert torch.equal(rank_columns(x, chunk=chunk), whole)


def test_spearman_columns_is_the_same_for_every_chunk_size():
    g = torch.Generator().manual_seed(0)
    target = torch.randn(12, generator=g)
    others = torch.randint(0, 5, (12, 7), generator=g).float()
    whole = spearman_columns(target, others)
    assert whole.abs().min() > 0
    for chunk in (1, 2, 3):
        assert torch.allclose(spearman_columns(target, others, chunk=chunk), whole)


def test_spearman_columns_of_two_points_is_plus_or_minus_one():
    rho = spearman_columns(torch.tensor([0.0, 1.0]), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert rho.tolist() == [1.0, -1.0]


def test_spearman_columns_of_a_constant_column_is_zero():
    rho = spearman_columns(torch.tensor([0.0, 1.0, 2.0]), torch.ones(3, 1))
    assert rho.tolist() == [0.0]


def test_spearman_columns_stays_within_minus_one_and_one():
    # Unclamped, float32 rounding puts this pair at 1 + 1.2e-7 and -1 - 1.2e-7.
    target = torch.arange(5_000).float()
    swapped = target.clone()
    swapped[[868, 869]] = swapped[[869, 868]]
    rho = spearman_columns(target, torch.stack([swapped, -swapped], dim=1))
    assert rho[0] <= 1.0
    assert rho[1] >= -1.0
    assert rho[0] == pytest.approx(1.0)
    assert rho[1] == pytest.approx(-1.0)
