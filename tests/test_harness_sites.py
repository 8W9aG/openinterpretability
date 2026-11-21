"""Dictionary site hooks: a read-only residual hook and an unknown site."""

import pytest
import torch

from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.sites import site_hooks
from tests.conftest import FakeTokenizer


@pytest.fixture
def runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def test_a_residual_hook_that_returns_nothing_leaves_the_forward_unchanged(runner):
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        plain = runner.model(ids).logits
    seen = []
    handles = site_hooks(runner, 1, "resid_post", lambda x, out: seen.append(x.shape))
    try:
        with torch.no_grad():
            hooked = runner.model(ids).logits
    finally:
        for h in handles:
            h.remove()
    assert seen == [torch.Size([1, 4, 32])]
    torch.testing.assert_close(hooked, plain)


def test_an_unknown_site_is_an_error(runner):
    with pytest.raises(ValueError, match="unknown dictionary site 'nowhere'"):
        site_hooks(runner, 0, "nowhere", lambda x, out: None)
