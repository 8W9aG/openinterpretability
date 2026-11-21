import torch

from openinterpretability.harness.confounds import ConfoundAccumulator


def test_an_empty_batch_adds_nothing_and_gives_an_empty_profile():
    acc = ConfoundAccumulator()
    empty = torch.zeros(0, dtype=torch.long)
    acc.update(empty, empty, torch.zeros(0))
    assert acc.token == {} and acc.position == {}
    profile = acc.finalize(None)
    assert (profile.token_r2, profile.position_r2, profile.frequency_correlation) == (0, 0, 0)
    assert profile.trigger_tokens == [] and profile.position_means == []
    assert profile.min_count == 3


def test_the_variance_explained_with_nothing_added_is_zero():
    # finalize returns early before this, so only a direct call reaches the guard.
    assert ConfoundAccumulator()._r2({}) == 0.0
