"""A neuron pole's values at the first position (harness/first_position.py)."""

import torch

from openinterpretability.harness import first_position
from openinterpretability.harness.first_position import NeuronReader
from openinterpretability.harness.hooks import pole_strength


def test_the_reader_gives_the_poles_strength_at_each_token(toy_runner):
    reader = NeuronReader(toy_runner, 1, 2, "negative")
    inputs = toy_runner.encode("abcd")
    want = toy_runner.neuron_activations(inputs, 1, 2, "negative")
    assert torch.equal(reader(inputs), want)


def test_a_damaged_cache_entry_is_computed_again(toy_runner, tmp_path, monkeypatch):
    monkeypatch.setenv(first_position.ENV, str(tmp_path))
    toy_runner.model_name = "toy"
    want = NeuronReader(toy_runner, 0, 3, "positive").first_position()
    (entry,) = tmp_path.glob("*.pt")
    entry.write_bytes(b"not a tensor")
    toy_runner.__dict__.pop("_first_position")  # a new run: nothing in memory
    again = NeuronReader(toy_runner, 0, 3, "positive").first_position()
    assert torch.equal(again, want)
    rewritten = torch.load(entry, weights_only=True)  # written again, whole
    assert torch.equal(pole_strength(rewritten, "positive"), want)
