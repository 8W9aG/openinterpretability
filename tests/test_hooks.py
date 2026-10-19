import pytest
import torch

from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.harness.hooks import (
    NeuronHookRunner,
    activation_histogram,
    isolate_pole,
    pole_strength,
    replace_pole,
)
from tests.conftest import D_MLP, N_LAYERS, VOCAB

IDS = torch.tensor([[1, 5, 7, 2, 9]])


def test_resolves_llama_style_layout(toy_model):
    runner = NeuronHookRunner(toy_model)
    assert runner.layer_path == "model.layers"
    assert runner.down_proj_path == "mlp.down_proj"
    assert runner.n_layers == N_LAYERS
    assert runner.mlp_width(0) == D_MLP


def test_resolves_gpt2_layout(tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2)
    assert runner.layer_path == "transformer.h"
    assert runner.down_proj_path == "mlp.c_proj"
    assert runner.mlp_width(1) == 4 * 32  # Conv1D weight is (in, out)
    assert runner.neuron_output_direction(1, 5).shape == (32,)


def test_unsupported_architecture_raises():
    with pytest.raises(ArchitectureNotSupportedError):
        NeuronHookRunner(torch.nn.Linear(2, 2))


def test_cache_captures_down_proj_input(toy_model):
    runner = NeuronHookRunner(toy_model)
    logits, cache = runner.run_with_cache(IDS, [0, 2])
    assert logits.shape == (1, 5, VOCAB)
    assert set(cache) == {0, 2}
    assert cache[0].shape == (1, 5, D_MLP)

    # The cached tensor is exactly act(up_proj(x)) at layer 0.
    mlp = toy_model.model.layers[0].mlp
    expected = mlp.act(mlp.up_proj(toy_model.embed(IDS)))
    torch.testing.assert_close(cache[0], expected)


def test_gpt2_cache_shape(tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2)
    _, cache = runner.run_with_cache(IDS, [0, 1])
    assert cache[1].shape == (1, 5, 128)


def test_hooks_are_removed(toy_model):
    runner = NeuronHookRunner(toy_model)
    baseline = runner.run_with_interventions(IDS)
    with runner.hooks({0: lambda a: torch.zeros_like(a)}):
        pass
    for layer in toy_model.model.layers:
        assert not layer.mlp.down_proj._forward_pre_hooks
    torch.testing.assert_close(runner.run_with_interventions(IDS), baseline)


def test_hooks_removed_after_exception(toy_model):
    runner = NeuronHookRunner(toy_model)

    def boom(_act):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        runner.run_with_interventions(IDS, {1: boom})
    assert not toy_model.model.layers[1].mlp.down_proj._forward_pre_hooks


def test_intervention_changes_output(toy_model):
    runner = NeuronHookRunner(toy_model)
    baseline = runner.run_with_interventions(IDS)
    ablated = runner.run_with_interventions(IDS, {1: lambda a: torch.zeros_like(a)})
    assert not torch.allclose(baseline, ablated)


def test_pole_isolation_partitions_activation():
    act = torch.tensor([-2.0, -0.5, 0.0, 0.3, 4.0])
    pos, neg = isolate_pole(act, "positive"), isolate_pole(act, "negative")
    torch.testing.assert_close(pos, torch.tensor([0.0, 0.0, 0.0, 0.3, 4.0]))
    torch.testing.assert_close(neg, torch.tensor([-2.0, -0.5, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(pos + neg, act)
    torch.testing.assert_close(
        pole_strength(act, "negative"), torch.tensor([2.0, 0.5, 0.0, 0.0, 0.0])
    )


def test_replace_pole_keeps_other_pole():
    target = torch.tensor([-1.0, 2.0, -3.0])
    source = torch.tensor([5.0, -4.0, 1.0])
    # Positive pole swapped: target's negative part kept, source's positive part injected.
    torch.testing.assert_close(
        replace_pole(target, source, "positive"), torch.tensor([4.0, 0.0, -2.0])
    )
    torch.testing.assert_close(
        replace_pole(target, source, "negative"), torch.tensor([0.0, -2.0, 0.0])
    )


def test_neuron_activations_pole_strength(toy_model):
    runner = NeuronHookRunner(toy_model)
    raw = runner.neuron_activations(IDS, 0, 3)
    pos = runner.neuron_activations(IDS, 0, 3, "positive")
    neg = runner.neuron_activations(IDS, 0, 3, "negative")
    assert raw.shape == (1, 5)
    assert (pos >= 0).all() and (neg >= 0).all()
    torch.testing.assert_close(pos - neg, raw)


def test_text_inputs_require_tokenizer(toy_model):
    with pytest.raises(ValueError, match="tokenizer"):
        NeuronHookRunner(toy_model).encode("hello")


def test_activation_histogram():
    hist = activation_histogram([0.0, 0.1, 0.2, 1.0], bins=4)
    assert sum(hist.counts) == 4
    assert len(hist.bin_edges) == 5
    assert activation_histogram([]).counts  # degenerate input still validates
