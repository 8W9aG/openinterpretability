"""Causal metric helpers in ``harness.causal``: unresolved claims and pairs of unequal length."""

from types import SimpleNamespace

import pytest
import torch

from openinterpretability.harness import causal
from openinterpretability.harness.heads import head_write_norms
from openinterpretability.harness.hooks import NeuronHookRunner, pole_strength
from tests.conftest import FakeTokenizer

IDENT = SimpleNamespace(layer=1, pole="positive", neuron_idx=3, head_idx=1)


@pytest.fixture
def runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def test_a_lookahead_claim_with_no_single_token_target_has_no_weights(runner):
    lookahead = SimpleNamespace(is_lookahead=True, target_upcoming_tokens=["ab", "cde"])
    assert causal.claimed_token_weights(SimpleNamespace(lookahead=lookahead), runner) is None


def test_neuron_changes_of_unequal_lengths_compare_the_last_tokens(runner):
    positions, position, change = causal._neuron_changes(runner, IDENT, "abcd", "abc")
    assert (positions, position) == ("last", -1)
    b, c = (
        pole_strength(runner.layer_activations(t, [1])[1][0], "positive") for t in ("abcd", "abc")
    )
    assert torch.allclose(change, (b[-1] - c[-1]).abs())


def test_head_changes_of_unequal_lengths_compare_the_last_tokens(runner):
    positions, position, change = causal._head_changes(runner, IDENT, "abcd", "abc")
    assert (positions, position) == ("last", -1)
    z_b, z_c = (runner.head_outputs(t, 1)[0].float() for t in ("abcd", "abc"))
    assert change.shape == (2,)
    assert torch.allclose(change, head_write_norms(runner, 1, z_b[-1] - z_c[-1]))
