import pytest
import torch

from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.jlens import transported_direction, vocabulary_lens
from tests.conftest import D_MODEL, N_LAYERS

IDS = torch.tensor([[1, 5, 7, 2, 9], [3, 3, 8, 1, 4]])


def exact_transport(runner, ids, layer, direction):
    """d/dε of Σ_t' h_final,t' with ε·direction added at every position, by autograd,
    divided by the number of (t ≤ t') pairs: the quantity the lens estimates."""
    eps = torch.zeros((), requires_grad=True)
    captured = []
    module = runner.final_norm() or runner.model.get_output_embeddings()
    handle = module.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    try:
        with (
            torch.enable_grad(),
            runner.hooks(output_interventions={layer: lambda out: out + eps * direction}),
        ):
            runner.forward(runner.encode(ids))
            total = captured[-1].sum(dim=(0, 1))  # [d_model]
            rows = [
                torch.autograd.grad(total[i], eps, retain_graph=True)[0] for i in range(len(total))
            ]
    finally:
        handle.remove()
    seq = ids.shape[1]
    pairs = ids.shape[0] * seq * (seq + 1) // 2
    return torch.stack(rows) / pairs


@pytest.mark.parametrize("layer", [0, 1])
def test_finite_difference_matches_autograd(toy_model, layer):
    runner = NeuronHookRunner(toy_model)
    direction = runner.neuron_output_direction(layer, 3)
    estimate = transported_direction(runner, IDS, layer, direction)
    exact = exact_transport(runner, IDS, layer, direction)
    torch.testing.assert_close(estimate, exact, atol=1e-4, rtol=1e-3)


def test_last_layer_jacobian_is_scaled_identity(toy_model):
    """Nothing follows the last MLP: J_{t,t'} is I when t' = t and 0 for later t'. The
    average over (t <= t') pairs is therefore I * 2/(seq+1), and both lenses agree."""
    runner = NeuronHookRunner(toy_model)
    direction = runner.neuron_output_direction(N_LAYERS - 1, 3)
    transported = transported_direction(runner, IDS, N_LAYERS - 1, direction)
    seq = IDS.shape[1]
    torch.testing.assert_close(transported, direction * 2 / (seq + 1), atol=1e-4, rtol=1e-3)


def test_vocabulary_lens(toy_runner):
    lens = vocabulary_lens(toy_runner, ["hello world", "abc"], N_LAYERS - 1, 3, "positive", k=4)
    assert lens.overlap == 1.0 and lens.direction_cosine == pytest.approx(1.0, abs=1e-4)
    assert lens.logit_promoted == lens.jacobian_promoted
    assert lens.prompts == 2
    for side in ("logit_promoted", "logit_suppressed", "jacobian_promoted", "jacobian_suppressed"):
        assert len(getattr(lens, f"{side}_z")) == 4
    assert lens.jacobian_promoted_z == sorted(lens.jacobian_promoted_z, reverse=True)
    assert lens.jacobian_promoted_z[0] > 0 > lens.jacobian_suppressed_z[0]
    assert lens.jacobian_suppressed_z == sorted(lens.jacobian_suppressed_z)

    early = vocabulary_lens(toy_runner, ["hello world", "abc"], 0, 3, "positive", k=4)
    assert -1 <= early.direction_cosine <= 1 and len(early.jacobian_promoted) == 4

    neg = vocabulary_lens(toy_runner, ["hello world"], N_LAYERS - 1, 3, "negative", k=4)
    assert neg.logit_promoted == lens.logit_suppressed  # the negative pole writes −w


def test_gpt2_uses_final_layernorm(tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2)
    assert runner.final_norm() is tiny_gpt2.transformer.ln_f
    direction = runner.neuron_output_direction(0, 7)
    estimate = transported_direction(runner, IDS, 0, direction)
    exact = exact_transport(runner, IDS, 0, direction)
    torch.testing.assert_close(estimate, exact, atol=1e-4, rtol=1e-3)
    assert estimate.shape == (32,) and D_MODEL == 8
