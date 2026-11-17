"""Wiring on a parallel-residual block (GPT-NeoX, as in Pythia; issue #3309)."""

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from openinterpretability.core.schemas import NeuronIdentifier
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.units import unit_of
from openinterpretability.harness.wiring import (
    _downstream_layers,
    _upstream_layers,
    attention_reads,
    parallel_residual,
    target_of,
    weight_wiring,
)
from tests.conftest import FakeTokenizer

HEADS, HEAD_DIM = 4, 8


def tiny_neox(parallel: bool = True) -> NeuronHookRunner:
    torch.manual_seed(0)
    config = GPTNeoXConfig(
        vocab_size=64, hidden_size=HEADS * HEAD_DIM, num_hidden_layers=4,
        num_attention_heads=HEADS, intermediate_size=64, max_position_embeddings=32,
        use_parallel_residual=parallel,
    )  # fmt: skip
    return NeuronHookRunner(GPTNeoXForCausalLM(config).eval(), FakeTokenizer())


def test_a_neox_block_with_parallel_residual_is_parallel():
    assert parallel_residual(tiny_neox(True))
    assert not parallel_residual(tiny_neox(False))


def test_a_parallel_neuron_has_no_upstream_heads_in_its_own_layer():
    runner = tiny_neox()
    t = target_of(
        runner, unit_of(NeuronIdentifier(model_name="m", layer=2, neuron_idx=3, pole="positive"))
    )
    assert _upstream_layers(t, True) == (range(2), range(2))
    assert _upstream_layers(t, False) == (range(2), range(3))  # sequential: its layer's heads
    wiring = weight_wiring(
        runner, unit_of(NeuronIdentifier(model_name="m", layer=2, neuron_idx=3, pole="positive"))
    )
    assert wiring.upstream_heads and max(e.layer for e in wiring.upstream_heads) == 1
    assert _downstream_layers(runner, t) == (range(3, 4), range(3, 4))


def test_neox_attention_reads_take_each_heads_interleaved_rows():
    runner = tiny_neox()
    block = runner.layers[1]
    reads = attention_reads(runner, 1)
    weight = block.attention.query_key_value.weight.detach().view(HEADS, 3, HEAD_DIM, -1)
    gain = block.input_layernorm.weight.detach()
    for i, channel in enumerate(("q", "k", "v")):
        expected = weight[2, i] * gain
        expected = expected - expected.mean(-1, keepdim=True)
        assert torch.allclose(reads[channel][2], expected, atol=1e-6)
