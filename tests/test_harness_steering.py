import pytest
import torch

from openinterpretability.core.schemas import NeuronIdentifier
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.steering import steering_samples
from openinterpretability.harness.units import NeuronUnit
from tests.conftest import N_LAYERS, FakeTokenizer

UNIT = NeuronUnit(
    NeuronIdentifier(model_name="toy", layer=N_LAYERS - 1, neuron_idx=4, pole="positive")
)


def greedy_ids(runner, ids, steps):
    seq = torch.tensor([ids])
    for _ in range(steps):
        seq = torch.cat([seq, runner.forward({"input_ids": seq})[:, -1:].argmax(-1)], dim=1)
    return seq[0, len(ids) :].tolist()


def test_a_continuation_stops_before_the_end_of_text_token(toy_model):
    runner = NeuronHookRunner(toy_model, FakeTokenizer())
    ids = runner.tokenizer.encode("abc")
    tokens = greedy_ids(runner, ids, 4)
    # Make the second generated token the end of text, so the text keeps only the first.
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = tokens[1]
    runner = NeuronHookRunner(toy_model, tokenizer)
    samples = steering_samples(runner, UNIT, 1.0, prompts=("abc",), max_new_tokens=4)
    first = tokens[: tokens.index(tokens[1])]
    assert samples.samples[0].baseline == tokenizer.decode(first)


def test_steering_samples_need_a_tokenizer(toy_model):
    with pytest.raises(ValueError, match="need a tokenizer"):
        steering_samples(NeuronHookRunner(toy_model), UNIT, 1.0)
