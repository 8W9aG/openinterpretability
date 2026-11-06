"""The helpers behind papers/why-neurons-rest/README.md."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from train_resting import (  # noqa: E402
    CONDITIONS,
    Activation,
    _layer_stats,
    build_model,
)


def test_fixed_rest_moves_each_neurons_mean_to_the_value():
    h = torch.randn(4, 16, 8) + 3
    out = Activation("tanh", -1.0, None)(h)
    centred = h - h.mean((0, 1), keepdim=True) - 1
    assert out.equal(torch.tanh(centred))


def test_learned_rest_is_a_parameter_of_the_model_for_each_neuron():
    model = build_model(CONDITIONS["rest_learned"], vocab_size=50, seed=0)
    names = [n for n, _ in model.named_parameters() if n.endswith("learned_rest")]
    assert len(names) == len(model.transformer.h)
    act = model.transformer.h[0].mlp.act
    assert act.learned_rest.shape == (model.config.n_inner,)
    with torch.no_grad():
        act.learned_rest.fill_(-2.0)
        h = torch.randn(4, 16, model.config.n_inner)
        centred = h - h.mean((0, 1), keepdim=True) - 2.0
        assert act(h).equal(torch.nn.functional.gelu(centred))


def test_baseline_activation_has_no_learned_rest():
    model = build_model(CONDITIONS["gelu"], vocab_size=50, seed=0)
    assert not [n for n, _ in model.named_parameters() if n.endswith("learned_rest")]


def test_mirrored_gelu_is_gelu_with_the_sign_of_h_reversed():
    h = torch.randn(32)
    out = Activation("mirrored_gelu", None, None)(h)
    assert out.equal(-torch.nn.functional.gelu(-h))


def test_layer_stats_splits_the_rest_into_offset_and_bias():
    model = build_model(CONDITIONS["gelu"], vocab_size=50, seed=0)
    mlp = model.transformer.h[0].mlp
    x = torch.ones(200, model.config.n_embd)
    stats = _layer_stats(mlp, x)
    weights, bias = mlp.c_fc.weight.detach(), mlp.c_fc.bias.detach()
    expected = (x[0] @ weights + bias).median().item()
    assert stats["median_h"] == pytest.approx(expected, abs=1e-5)
    assert stats["median_offset"] == pytest.approx((x[0] @ weights).median().item(), abs=1e-5)
    assert stats["median_learned_rest"] is None
