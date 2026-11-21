"""A unit's wiring by attribution (openinterpretability/harness/wiring_activations.py)."""

import pytest

from openinterpretability.core.exceptions import ArchitectureNotSupportedError
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.wiring_activations import activation_wiring
from tests.conftest import FakeTokenizer
from tests.test_wiring import head


def test_a_head_needs_attention_that_the_first_layer_supports_too(tiny_gpt2):
    # Without layer 0's q/k/v projection, the model's attention is not supported, but the
    # head on layer 1 still has a target.
    del tiny_gpt2.transformer.h[0].attn.c_attn
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    with pytest.raises(ArchitectureNotSupportedError, match="a head's wiring needs supported"):
        activation_wiring(runner, head(1, 0), ["hello"], threshold=0.0)
