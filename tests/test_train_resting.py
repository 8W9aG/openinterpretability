"""The helpers behind papers/why-neurons-rest/README.md."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from train_resting import (  # noqa: E402
    CONDITIONS,
    INIT_MEAN_GELU,
    INIT_MEAN_RELU,
    LEAKY_SLOPE,
    SGD_MOMENTUM,
    Activation,
    PushRecorder,
    _layer_stats,
    build_model,
    make_optimizer,
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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("softplus", torch.nn.functional.softplus),
        ("silu", torch.nn.functional.silu),
        ("leaky_relu", lambda h: torch.nn.functional.leaky_relu(h, LEAKY_SLOPE)),
    ],
)
def test_each_new_activation_is_the_torch_function(name, expected):
    h = torch.randn(32)
    assert Activation(name, None, None)(h).equal(expected(h))


def test_sgd_condition_uses_momentum_and_its_own_learning_rate():
    condition = CONDITIONS["sgd"]
    optimizer = make_optimizer(condition, build_model(condition, vocab_size=50, seed=0))
    assert isinstance(optimizer, torch.optim.SGD)
    group = optimizer.param_groups[0]
    assert group["lr"] == condition.learning_rate
    assert group["momentum"] == SGD_MOMENTUM


def test_learned_rest_rate_is_a_multiple_of_the_conditions_rate():
    condition = CONDITIONS["rest_learned_fast"]
    optimizer = make_optimizer(condition, build_model(condition, vocab_size=50, seed=0))
    assert isinstance(optimizer, torch.optim.AdamW)
    others, rests = optimizer.param_groups
    assert rests["lr"] == pytest.approx(others["lr"] * condition.rest_lr_factor)


def _one_recorded_step(name: str):
    """Return a small model and its recorder after one training step of condition ``name``."""
    condition = CONDITIONS[name]
    model = build_model(condition, vocab_size=50, seed=0).train()
    optimizer = make_optimizer(condition, model)
    recorder = PushRecorder(model)
    batch = torch.randint(50, (2, 16), generator=torch.Generator().manual_seed(0))
    model(batch, labels=batch).loss.backward()
    recorder.before_step()
    return model, optimizer, recorder


def test_push_on_the_rest_is_the_gradient_on_the_bias_with_its_sign_reversed():
    model, optimizer, recorder = _one_recorded_step("gelu")
    expected = (-model.transformer.h[0].mlp.c_fc.bias.grad).median().item()
    optimizer.step()
    recorder.after_step()
    for block in model.transformer.h:
        assert block.mlp.act.rest_grad.allclose(block.mlp.c_fc.bias.grad, atol=1e-7)
    assert recorder.row()[0]["median_push"] == pytest.approx(expected, abs=1e-7)


def test_movement_is_the_change_in_h_at_the_mean_input():
    model, optimizer, recorder = _one_recorded_step("gelu")
    mlp = model.transformer.h[1].mlp
    x_mean = recorder.inputs[1].clone()
    weight, bias = mlp.c_fc.weight.detach().clone(), mlp.c_fc.bias.detach().clone()
    optimizer.step()
    recorder.after_step()
    moved = (x_mean @ mlp.c_fc.weight.detach() + mlp.c_fc.bias.detach()) - (x_mean @ weight + bias)
    assert recorder.row()[1]["median_moved"] == pytest.approx(moved.median().item(), abs=1e-6)


def test_weights_push_is_the_weight_gradient_along_the_mean_input():
    model, optimizer, recorder = _one_recorded_step("gelu")
    mlp = model.transformer.h[2].mlp
    expected = (-(recorder.inputs[2] @ mlp.c_fc.weight.grad)).median().item()
    optimizer.step()
    recorder.after_step()
    assert recorder.row()[2]["median_push_weights"] == pytest.approx(expected, abs=1e-7)


def test_leaky_relu_condition_sets_its_own_slope():
    model = build_model(CONDITIONS["leaky_relu_0.01"], vocab_size=50, seed=0)
    h = torch.randn(32)
    out = model.transformer.h[0].mlp.act(h)
    assert out.equal(torch.nn.functional.leaky_relu(h, 0.01))


def test_output_shift_adds_a_constant_and_leaves_the_slope():
    h = torch.randn(4, 16, 8)
    act = Activation("relu", None, None, output_shift=-0.1)
    assert act(h).allclose(torch.nn.functional.relu(h) - 0.1)
    assert act._slope(h).equal((h > 0).float())


def test_init_means_are_those_of_relu_and_gelu_at_an_sd_of_0_31():
    h = torch.randn(1_000_000, generator=torch.Generator().manual_seed(0)) * 0.31
    assert torch.nn.functional.relu(h).mean().item() == pytest.approx(INIT_MEAN_RELU, abs=0.002)
    assert torch.nn.functional.gelu(h).mean().item() == pytest.approx(INIT_MEAN_GELU, abs=0.002)


def test_centred_output_has_no_output_push():
    free, _, _ = _one_recorded_step("gelu")
    model, optimizer, recorder = _one_recorded_step("gelu_output_centred")
    for centred, block in zip(model.transformer.h, free.transformer.h, strict=True):
        # Only rounding is left: a millionth of the uncentred output's push, or less.
        scale = block.mlp.act.output_grad.abs().max()
        assert centred.mlp.act.output_grad.abs().max() < 1e-6 * scale
    optimizer.step()
    recorder.after_step()
    row = recorder.row()[1]
    assert abs(row["median_push_slope"]) < 1e-3 * abs(row["median_push"])


def test_slope_and_spread_terms_add_up_to_the_push_on_each_neuron():
    model, optimizer, recorder = _one_recorded_step("relu")
    act = model.transformer.h[1].mlp.act
    push, output_push = -act.rest_grad, -act.output_grad
    slope_term = output_push * act.mean_slope
    optimizer.step()
    recorder.after_step()
    window = recorder.window
    assert window["push_slope"][1].allclose(slope_term)
    assert (window["push"][1] - window["push_slope"][1]).allclose(push - slope_term)


def test_energy_push_and_output_correlation_of_an_l2_loss():
    act = Activation("softplus", None, None)
    act.record = True
    h = torch.randn(2, 16, 8, requires_grad=True)
    out = act(h)
    (out.square().sum() / 2).backward()  # then dL/da is a on each token
    assert act.output_correlation.allclose(torch.ones(8), atol=1e-5)
    expected = -(out * torch.sigmoid(h)).sum((0, 1)).detach()
    assert act.energy_push.allclose(expected, atol=1e-5)
    assert (-act.rest_grad).allclose(act.energy_push, atol=1e-5)


@pytest.mark.parametrize(
    ("name", "final"), [("relu_norms_centred", False), ("relu_all_norms_centred", True)]
)
def test_centred_norms_ignore_a_constant_offset_in_their_input(name, final):
    model = build_model(CONDITIONS[name], vocab_size=50, seed=0)
    norm = model.transformer.h[0].ln_1
    x = torch.randn(2, 16, model.config.n_embd)
    assert norm(x + 3.0).allclose(norm(x), atol=1e-5)
    assert hasattr(model.transformer.ln_f, "norm") == final


def test_final_norm_condition_centres_only_the_final_norm():
    model = build_model(CONDITIONS["relu_final_norm_centred_dropout_0"], vocab_size=50, seed=0)
    assert not hasattr(model.transformer.h[0].ln_1, "norm")
    assert hasattr(model.transformer.ln_f, "norm")


@pytest.mark.parametrize(
    "name", ["dropout_0", "relu_dropout_0", "relu_all_norms_centred_dropout_0"]
)
def test_no_dropout_conditions_turn_off_every_dropout(name):
    config = build_model(CONDITIONS[name], vocab_size=50, seed=0).config
    assert config.resid_pdrop == config.embd_pdrop == config.attn_pdrop == 0.0


def test_mlp_output_bias_gets_no_gradient_when_no_route_sees_a_constant():
    # Every norm centres its input over tokens and nothing drops out, so a constant added to
    # the residual stream has no effect on the loss.
    model = build_model(CONDITIONS["relu_all_norms_centred_dropout_0"], vocab_size=50, seed=0)
    ids = torch.randint(50, (2, 16))
    model(ids, labels=ids).loss.backward()
    bias_grad = model.transformer.h[0].mlp.c_proj.bias.grad
    weight_grad = model.transformer.h[0].mlp.c_proj.weight.grad
    assert bias_grad.abs().max() < 1e-6 * weight_grad.abs().max().clamp_min(1.0)


def test_gelu_new_is_the_tanh_form_that_gpt2_uses():
    from transformers.activations import NewGELUActivation

    h = torch.randn(64)
    assert Activation("gelu_new", None, None)(h).allclose(NewGELUActivation()(h), atol=1e-6)
