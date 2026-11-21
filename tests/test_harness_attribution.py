import pytest
import torch

from openinterpretability.core.exceptions import ShapeMismatchError
from openinterpretability.harness.attribution import attribution_patching
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import D_MLP

CLEAN = torch.tensor([[1, 5, 7, 2, 9]])
SHORT = torch.tensor([[1, 5, 3]])


@pytest.fixture
def runner(toy_model):
    return NeuronHookRunner(toy_model)


def test_last_position_attribution_takes_prompts_of_different_lengths(runner):
    scores = attribution_patching(runner, CLEAN, SHORT, [1, 2], positions="last")
    assert set(scores) == {1, 2}
    assert scores[2].shape == (D_MLP,) and torch.isfinite(scores[2]).all()


def test_attribution_over_all_positions_refuses_prompts_of_different_lengths(runner):
    with pytest.raises(ShapeMismatchError, match="use positions='last'"):
        attribution_patching(runner, CLEAN, SHORT, [1])
