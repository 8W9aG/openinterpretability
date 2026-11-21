"""NeuronHookRunner's checks on the model it is given (openinterpretability/harness/hooks.py)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openinterpretability.core.exceptions import ArchitectureNotSupportedError
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import N_LAYERS, FakeTokenizer, ToyModel


def test_a_tokenizer_without_a_pad_token_pads_with_its_end_token(toy_model):
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
    NeuronHookRunner(toy_model, tokenizer)
    assert tokenizer.pad_token == "<eos>"


def test_layers_without_a_known_down_projection_are_refused(toy_model):
    for block in toy_model.model.layers:
        block.mlp = nn.Identity()
    with pytest.raises(ArchitectureNotSupportedError, match=r"down projection under model\.layers"):
        NeuronHookRunner(toy_model)


def test_the_device_is_that_of_the_first_weight_not_left_on_disk(toy_model):
    toy_model.embed.to("meta")  # the first weight is left on disk, as a streamed layer's
    assert NeuronHookRunner(toy_model).device == torch.device("cpu")
    assert next(toy_model.parameters()).device.type == "meta"
    all_meta = ToyModel().to("meta")
    assert NeuronHookRunner(all_meta).device == torch.device("cpu")  # no real weight at all


def test_a_layer_out_of_range_is_an_index_error(toy_model):
    runner = NeuronHookRunner(toy_model)
    with pytest.raises(IndexError, match=f"layer {N_LAYERS} out of range for {N_LAYERS}-layer"):
        runner.down_proj(N_LAYERS)
    with pytest.raises(IndexError, match="layer -1 out of range"):
        runner.attn_out(-1)


def test_an_attention_width_that_the_heads_do_not_divide_is_refused(tiny_gpt2):
    tiny_gpt2.config.n_head = 3  # the output projection is 32 wide
    with pytest.raises(ArchitectureNotSupportedError, match="32 is not a multiple of 3 heads"):
        NeuronHookRunner(tiny_gpt2).head_dim(0)


def test_a_layer_without_a_pre_mlp_norm_is_refused(toy_model):
    with pytest.raises(ArchitectureNotSupportedError, match=r"norm before the MLP.*\[1\]"):
        NeuronHookRunner(toy_model).mlp_norm(1)


def test_a_down_projection_outside_an_mlp_module_has_no_mlp(toy_model):
    for block in toy_model.model.layers:
        block.proj = block.mlp.down_proj
    runner = NeuronHookRunner(toy_model, down_proj_path="proj")
    assert runner.down_proj(0) is toy_model.model.layers[0].mlp.down_proj
    with pytest.raises(ArchitectureNotSupportedError, match="proj is not inside an MLP module"):
        runner.mlp(0)


class _NoAttentions(ToyModel):
    """A model that takes ``output_attentions`` but returns no attention weights."""

    def forward(self, input_ids, attention_mask=None, **_):
        return super().forward(input_ids, attention_mask)


def test_attention_patterns_need_a_model_that_returns_them():
    runner = NeuronHookRunner(_NoAttentions().eval(), FakeTokenizer())
    with pytest.raises(ArchitectureNotSupportedError, match='attn_implementation="eager"'):
        runner.attention_patterns("abc", 0)
