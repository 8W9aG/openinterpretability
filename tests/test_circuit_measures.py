"""Circuit measures on neurons and heads (openinterp/harness/circuit_measures.py, #4284)."""

import pytest

from openinterp.harness import circuit_measures as cm
from openinterp.harness.hooks import NeuronHookRunner
from tests.conftest import VOCAB, FakeTokenizer


def _task():
    clean = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]]
    corrupted = [[9, 2, 3, 4], [10, 6, 7, 8], [11, 4, 6, 8]]
    return cm.PairTask(clean, corrupted, good=[3, 7, 12], bad=[13, 14, 15])


@pytest.fixture
def runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def _everything(runner):
    units = [cm.Member("N", layer, i) for layer in range(runner.n_layers)
             for i in range(runner.mlp_width(layer))]  # fmt: skip
    units += [cm.Member("H", layer, i) for layer in range(runner.n_layers)
              for i in range(runner.n_heads)]  # fmt: skip
    return units


def test_members_parse_from_slugs_and_pages():
    assert cm.Member.parse("L4/N2999") == cm.Member("N", 4, 2999)
    assert cm.Member.parse("models/m/L7/H2.html") == cm.Member("H", 7, 2)
    assert str(cm.Member("H", 7, 2)) == "L7/H2"
    with pytest.raises(ValueError):
        cm.Member.parse("L1/F3")


def test_keeping_everything_keeps_the_whole_metric(runner):
    task = _task()
    means = cm.task_means(runner, task)
    every = cm.measures(runner, task, means, _everything(runner))
    assert every["faithfulness"] == pytest.approx(1, abs=1e-4)
    assert every["necessity"] == pytest.approx(1, abs=1e-4)
    none = cm.measures(runner, task, means, [])
    assert none["faithfulness"] == pytest.approx(0, abs=1e-4)
    assert none["necessity"] == pytest.approx(0, abs=1e-4)


def test_ablating_a_head_changes_the_metric(runner):
    task = _task()
    means = cm.task_means(runner, task)
    heads = [cm.Member("H", 0, 0), cm.Member("H", 1, 1)]
    assert cm.ablated(runner, task, means, heads) != pytest.approx(cm.ablated(runner, task, means))


def test_attribution_scores_every_neuron_and_head(runner):
    scores = cm.attribution(runner, _task())
    assert len(scores) == len(_everything(runner))
    assert any(abs(v) > 0 for v in scores.values())


def test_grow_keeps_the_seed_then_adds_the_strongest():
    scores = {cm.Member("N", 0, i): float(i) for i in range(5)}
    seed = [cm.Member("H", 1, 0)]
    assert cm.grow(seed, scores, 3) == [seed[0], cm.Member("N", 0, 4), cm.Member("N", 0, 3)]


def test_random_like_keeps_layers_and_kinds(runner):
    import random

    members = [cm.Member("N", 1, 3), cm.Member("H", 0, 1)]
    other = cm.random_like(runner, members, random.Random(0))
    assert [(m.kind, m.layer) for m in other] == [("N", 1), ("H", 0)]
    assert all(m.index < VOCAB * 100 for m in other)
