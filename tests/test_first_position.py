import torch

from openinterp.harness import first_position
from openinterp.harness.first_position import NeuronReader, strengths_of
from openinterp.harness.hooks import pole_strength
from openinterp.harness.scan import first_position_share


def _passes(runner, monkeypatch) -> list[int]:
    """Count the vocabulary passes, by counting calls into the model."""
    calls = []
    real = runner.neuron_activations
    monkeypatch.setattr(
        runner, "neuron_activations", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    return calls


def test_a_neurons_first_positions_match_the_direct_pass(toy_runner):
    for pole in ("positive", "negative"):
        reader = NeuronReader(toy_runner, 0, 3, pole)
        direct = strengths_of(
            toy_runner, lambda x, p=pole: toy_runner.neuron_activations(x, 0, 3, p)
        )
        assert torch.equal(reader.first_position(), direct)
        line = float(direct.median())
        assert first_position_share(toy_runner, reader, line) == float(
            (direct >= line).float().mean()
        )


def test_both_poles_share_one_pass_of_the_vocabulary(toy_runner, monkeypatch):
    calls = _passes(toy_runner, monkeypatch)
    pos = NeuronReader(toy_runner, 0, 3, "positive").first_position()
    neg = NeuronReader(toy_runner, 0, 3, "negative").first_position()
    assert len(calls) == 1
    raw = strengths_of(toy_runner, lambda x: toy_runner.neuron_activations(x, 0, 3))
    assert torch.equal(pos, pole_strength(raw, "positive"))
    assert torch.equal(neg, pole_strength(raw, "negative"))


def test_a_later_run_reads_the_values_from_disk(toy_model, monkeypatch):
    from openinterp.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    first = NeuronHookRunner(toy_model, FakeTokenizer())
    first.model_name = "toy"
    want = NeuronReader(first, 1, 2, "positive").first_position()
    again = NeuronHookRunner(toy_model, FakeTokenizer())  # a new process: nothing in memory
    again.model_name = "toy"
    calls = _passes(again, monkeypatch)
    assert torch.equal(NeuronReader(again, 1, 2, "positive").first_position(), want)
    assert calls == []
    monkeypatch.setenv(first_position.ENV, "off")
    off = NeuronHookRunner(toy_model, FakeTokenizer())
    off.model_name = "toy"
    calls = _passes(off, monkeypatch)
    NeuronReader(off, 1, 2, "positive").first_position()
    assert calls == [1]
