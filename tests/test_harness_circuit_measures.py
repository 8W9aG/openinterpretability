"""Circuit measures: refused tasks and growing a member set."""

import pytest

from openinterpretability.harness import circuit_measures as cm
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import FakeTokenizer


def test_a_pair_of_different_lengths_is_refused():
    with pytest.raises(ValueError, match="different token counts"):
        cm.PairTask([[1, 2, 3]], [[1, 2]], good=[3], bad=[4])


def test_a_task_whose_metric_never_moves_is_refused(tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    # The same target on both sides makes the metric 0 under every ablation.
    task = cm.PairTask([[1, 2, 3, 4]], [[9, 2, 3, 4]], good=[5], bad=[5])
    means = cm.task_means(runner, task)
    with pytest.raises(ValueError, match="doesn't change when every unit is ablated"):
        cm.measures(runner, task, means, [cm.Member("N", 0, 1)])


def test_grow_takes_every_score_when_there_are_too_few_and_skips_the_seed():
    a, b = cm.Member("N", 0, 1), cm.Member("H", 1, 0)
    grown = cm.grow([a], {a: 5.0, b: -1.0}, size=10)
    assert grown == [a, b]
