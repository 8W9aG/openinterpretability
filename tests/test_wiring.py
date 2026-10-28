"""Wiring from weights and from activations, on toy, GPT-2 and Qwen2 models (no download)."""

import pytest
import torch

from openinterp.core.schemas import NeuronIdentifier
from openinterp.harness.hooks import NeuronHookRunner, ablate_pole, pole_strength
from openinterp.harness.wiring import activation_wiring, attention_reads, weight_wiring
from tests.conftest import N_LAYERS, FakeTokenizer

PROMPTS = ["hello world", "abcabc", "the cat sat"]


def neuron(layer: int, idx: int, pole: str = "positive") -> NeuronIdentifier:
    return NeuronIdentifier(model_name="toy", layer=layer, neuron_idx=idx, pole=pole)


@pytest.fixture
def gpt2_runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


@pytest.fixture
def qwen_runner():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return NeuronHookRunner(Qwen2ForCausalLM(config).eval(), FakeTokenizer())


def test_weights_match_virtual_weights_by_hand(gpt2_runner, tiny_gpt2):
    block0, block1 = tiny_gpt2.transformer.h
    wiring = weight_wiring(gpt2_runner, neuron(1, 5), k=3)
    read = block1.mlp.c_fc.weight[:, 5].detach() * block1.ln_2.weight.detach()
    read = read - read.mean()
    expected = block0.mlp.c_proj.weight.detach() @ (read / read.norm())  # every L0 neuron
    top = wiring.upstream_neurons[0]
    assert (top.kind, top.layer, top.channels.keys()) == ("mlp_neuron", 0, {"in"})
    assert top.index == int(expected.abs().argmax())
    assert top.weight == pytest.approx(float(expected[top.index]), abs=1e-5)
    assert len(wiring.upstream_heads) == 3 and all(e.weight >= 0 for e in wiring.upstream_heads)
    assert {e.layer for e in wiring.upstream_heads} <= {0, 1}
    assert wiring.downstream_neurons == [] and wiring.downstream_heads == []


def test_negative_pole_flips_downstream_signs_only(gpt2_runner):
    pos = weight_wiring(gpt2_runner, neuron(0, 3, "positive"))
    neg = weight_wiring(gpt2_runner, neuron(0, 3, "negative"))
    assert [e.weight for e in neg.downstream_neurons] == [-e.weight for e in pos.downstream_neurons]
    assert neg.downstream_heads == pos.downstream_heads and neg.upstream_heads == pos.upstream_heads
    assert neg.upstream_neurons == pos.upstream_neurons  # signed by the read, not the pole


def test_gated_mlp_and_grouped_query_attention(qwen_runner):
    reads = attention_reads(qwen_runner, 1)
    assert reads["q"].shape == (4, 8, 32) and reads["k"].shape == (4, 8, 32)
    assert torch.equal(reads["k"][0], reads["k"][1]) and not torch.equal(
        reads["q"][0], reads["q"][1]
    )
    wiring = weight_wiring(qwen_runner, neuron(1, 7))
    edge = wiring.upstream_neurons[0]
    assert edge.channels.keys() == {"gate", "up"}
    assert edge.weight == max(edge.channels.values(), key=abs)
    assert {e.channels.keys() == {"q", "k", "v"} for e in wiring.downstream_heads} == {True}
    assert len({e.index for e in wiring.downstream_heads if e.layer == 2}) == 4


def test_activation_downstream_is_the_direct_effect_of_ablating_the_pole(toy_runner):
    # The toy model has no attention, so a later layer sees the pole only directly.
    ident = neuron(0, 2, "negative")
    wiring = activation_wiring(toy_runner, ident, ["hello"], threshold=0.0, k=50)
    assert wiring.prompts == 1 and wiring.upstream_heads is None
    _, clean = toy_runner.run_with_cache("hello", [0, 1])
    t = int(pole_strength(clean[0][0, :, 2], "negative").argmax())
    with toy_runner.hooks({0: ablate_pole(2, "negative")}):
        _, ablated = toy_runner.run_with_cache("hello", [1])
    effect = clean[1][0, t] - ablated[1][0, t]
    for e in (e for e in wiring.downstream_neurons if e.layer == 1):
        assert e.weight == pytest.approx(float(effect[e.index]), abs=1e-5)


def test_activation_upstream_is_first_order_in_each_write(toy_runner):
    ident = neuron(N_LAYERS - 1, 4)
    wiring = activation_wiring(toy_runner, ident, ["hello"], threshold=0.0, k=100)
    edge = next(e for e in wiring.upstream_neurons if e.layer == N_LAYERS - 2)

    def strength(scale: float) -> float:
        def shrink(act):
            act = act.clone()
            act[..., edge.index] *= scale
            return act

        # Scaling the neuron at every position changes later layers only through this one.
        with toy_runner.hooks({edge.layer: shrink}):
            return toy_runner.max_pole_strength("hello", ident.layer, 4, "positive")

    eps = 1e-3
    assert (strength(1.0) - strength(1 - eps)) / eps == pytest.approx(
        edge.weight, rel=1e-2, abs=1e-5
    )


def test_activation_wiring_on_gpt2_and_inactive_prompts(gpt2_runner):
    wiring = activation_wiring(gpt2_runner, neuron(1, 5), PROMPTS, threshold=0.0)
    assert wiring.prompts == len(PROMPTS) and wiring.downstream_heads is None
    assert len(wiring.upstream_heads) == 4 and len(wiring.upstream_neurons) == 10
    assert activation_wiring(gpt2_runner, neuron(1, 5), PROMPTS, threshold=1e9) is None
