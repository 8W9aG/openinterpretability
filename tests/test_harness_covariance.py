"""Partner poles that co-activate with the target (harness/covariance.py)."""

import torch

from openinterpretability.harness.covariance import PartnerSearch, _top_tokens


def test_fewer_than_three_sampled_tokens_give_no_partners():
    search = PartnerSearch(4)
    search.update(torch.tensor([1.0, 2.0]), torch.ones(2, 4))
    assert search.partners(exclude=0, active_threshold=0.5) == []


def test_a_target_that_never_fires_ranks_partners_by_rank_correlation_alone():
    gen = torch.Generator().manual_seed(0)
    acts = torch.randn(200, 4, generator=gen)
    target = acts[:, 2].clamp(min=0)  # moves with neuron 2's positive pole
    search = PartnerSearch(4)
    search.update(target, acts)
    partners = search.partners(exclude=0, active_threshold=1e9, k=3)  # never reached
    assert all(p.coactivation_lift == 1.0 and p.coactivation_lift_lower == 0.0 for p in partners)
    assert all(p.coactive_tokens == 0 for p in partners)
    assert (partners[0].neuron_idx, partners[0].pole) == (2, "positive")


def test_top_tokens_stop_when_the_tokens_run_out():
    strength = torch.tensor([3.0, 2.0, 1.0])
    ids = torch.tensor([7, 7, 9])
    assert _top_tokens(strength, -1.0, ids, lambda t: f"<{t[0]}>") == ["<7>", "<9>"]
