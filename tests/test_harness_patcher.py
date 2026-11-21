"""Activation patching: the logit-difference metric and refused prompt shapes."""

from types import SimpleNamespace

import pytest
import torch

from openinterpretability.core.exceptions import ShapeMismatchError
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.patcher import (
    ActivationPatcher,
    logit_diff_metric,
    metric_change,
    token_logit_metric,
)

CLEAN = torch.tensor([[1, 5, 7, 2, 9]])
CORRUPT = torch.tensor([[1, 5, 3, 2, 4]])


@pytest.fixture
def runner(toy_model):
    return NeuronHookRunner(toy_model)


def test_logit_diff_is_the_batch_mean_of_one_logit_minus_another():
    logits = torch.zeros(2, 3, 5)
    logits[0, -1, 1], logits[0, -1, 2] = 4.0, 1.0
    logits[1, -1, 1], logits[1, -1, 2] = 2.0, 3.0
    assert float(logit_diff_metric(1, 2)(logits)) == pytest.approx(1.0)


def test_patch_many_takes_one_source_and_one_target(runner):
    effect = metric_change(token_logit_metric(3))
    two = torch.cat([CLEAN, CORRUPT])
    with pytest.raises(ValueError, match="one source and one target prompt"):
        ActivationPatcher(runner).patch_many(two, CORRUPT, 1, [0], "positive", effect=effect)


def test_patch_many_at_every_position_needs_prompts_of_one_length(runner):
    effect = metric_change(token_logit_metric(3))
    with pytest.raises(ShapeMismatchError, match="differ in sequence length"):
        ActivationPatcher(runner).patch_many(
            CLEAN, CORRUPT[:, :3], 1, [0], "positive", effect=effect
        )


def test_resample_ablation_scores_every_source_with_a_given_metric(runner):
    metric = token_logit_metric(3)
    pole = SimpleNamespace(layer=1, neuron_idx=4, pole="positive")
    patcher = ActivationPatcher(runner)
    result = patcher.resample_ablation(CLEAN, [CORRUPT], pole, metric=metric)
    expected = patcher.patch(CORRUPT, CLEAN, pole, metric=metric)
    (only,) = result.per_source
    assert only.metric_delta == pytest.approx(expected.metric_delta)
    assert only.baseline_metric == pytest.approx(
        float(metric(runner.run_with_interventions(CLEAN)))
    )
