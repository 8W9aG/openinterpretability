"""Integrated gradients at a pole's peak (harness/input_attribution.py)."""

import pytest

from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.input_attribution import input_attribution
from tests.conftest import FakeTokenizer
from tests.test_input_attribution import TEXT, example, peak_pole


def test_steps_double_until_the_cap_when_the_sum_never_converges(tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    pole = peak_pole(runner, 1, 7)
    examples = [example(runner, t, 1, 7, pole) for t in (TEXT, "hgfedcba", "aabbccdd")]
    result = input_attribution(runner, examples, 1, 7, pole, steps=1, max_steps=4, tolerance=1e-9)
    assert result.examples  # some examples have a gap to attribute
    assert all(r.steps == 4 and not r.converged for r in result.examples)  # 1, then 2, then 4


def test_attribution_needs_a_tokenizer(toy_model):
    runner = NeuronHookRunner(toy_model, None)
    with pytest.raises(ValueError, match="input_attribution requires a tokenizer"):
        input_attribution(runner, [], 0, 0, "positive")
