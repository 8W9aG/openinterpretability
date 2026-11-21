from types import SimpleNamespace

import pytest
import torch

from openinterpretability.core.exceptions import ArchitectureNotSupportedError
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.patchscope import patchscope
from tests.conftest import N_LAYERS, FakeTokenizer, ToyModel

PROMPTS = {"short": "ab{x}c"}


class BlocklessModel(ToyModel):
    """Runs each block's MLP itself, so no block runs as a module."""

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for block in self.model.layers:
            x = x + block.mlp(x)
        return SimpleNamespace(logits=self.lm_head(x))


def test_a_block_that_never_runs_as_a_module_is_not_supported():
    torch.manual_seed(0)
    runner = NeuronHookRunner(BlocklessModel().eval(), FakeTokenizer())
    direction = runner.neuron_output_direction(1, 3)
    with pytest.raises(ArchitectureNotSupportedError, match="layer 1's block never ran"):
        patchscope(runner, 1, direction, prompts=PROMPTS)


def test_every_row_stops_at_the_end_of_text_token(toy_model):
    with torch.no_grad():
        toy_model.lm_head.weight.zero_()  # every logit is 0, so greedy picks token 0
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = 0
    runner = NeuronHookRunner(toy_model, tokenizer)
    direction = runner.neuron_output_direction(N_LAYERS - 1, 3)
    readout = patchscope(runner, N_LAYERS - 1, direction, prompts=PROMPTS, max_new_tokens=5)
    assert len(readout.readings) == 3  # one layer, three scales
    assert all(r.text == "" and r.control_text == "" for r in readout.readings)


def test_a_patchscope_needs_a_tokenizer(toy_model):
    with pytest.raises(ValueError, match="needs a tokenizer"):
        patchscope(NeuronHookRunner(toy_model), 1, torch.ones(8))
