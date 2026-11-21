"""EPO dreaming of a neuron pole (openinterpretability/harness/dreaming.py, issue #3228)."""

import torch

from openinterpretability.harness.dreaming import _score, dream
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)


def _objective(runner, tokens: list[int], lam: float) -> float:
    ids = torch.tensor([tokens])
    table = runner.model.get_input_embeddings().weight
    with torch.no_grad():
        f, ce = _score(runner, 1, 5, "positive", table[ids], ids)
    return float(f[0] - lam * ce[0])


def test_each_member_ends_at_least_as_good_as_its_start(qwen_runner):  # noqa: F811
    lambdas = (0.0, 1.0)
    start = dream(qwen_runner, 1, 5, "positive", length=6, lambdas=lambdas, steps=0, seed=3)
    end = dream(qwen_runner, 1, 5, "positive", length=6, lambdas=lambdas, children=8,
                top_k=8, steps=10, seed=3)  # fmt: skip
    for s, e in zip(start, end, strict=True):
        assert len(e.tokens) == 6
        assert (
            _objective(qwen_runner, e.tokens, e.lam)
            >= _objective(qwen_runner, s.tokens, s.lam) - 1e-5
        )


def test_a_larger_lambda_gives_text_no_less_fluent(qwen_runner):  # noqa: F811
    low, high = dream(qwen_runner, 1, 5, "positive", length=6, lambdas=(0.0, 20.0), children=8,
                      top_k=8, steps=15, seed=1)  # fmt: skip
    assert high.cross_entropy <= low.cross_entropy + 1e-5
    assert low.strength >= high.strength - 1e-5


def test_min_lowers_the_pole_and_keeps_frozen_tokens(qwen_runner):  # noqa: F811
    # From a high dream, `min` silences the pole while the frozen tokens stay put (#3779).
    (high,) = dream(qwen_runner, 1, 5, "positive", length=6, lambdas=(0.0,), children=8,
                    top_k=8, steps=10, seed=2)  # fmt: skip
    (low,) = dream(qwen_runner, 1, 5, "positive", lambdas=(0.0,), children=8, top_k=8,
                   steps=10, seed=2, goal="min", start=high.tokens, frozen=(0, 1))  # fmt: skip
    assert low.tokens[:2] == high.tokens[:2] and len(low.tokens) == 6
    assert low.strength <= high.strength + 1e-5


def test_hit_moves_toward_the_target(qwen_runner):  # noqa: F811
    (start,) = dream(qwen_runner, 1, 5, "positive", length=6, lambdas=(0.0,), steps=0, seed=4)
    target = start.strength + 1.0
    (end,) = dream(qwen_runner, 1, 5, "positive", lambdas=(0.0,), children=8, top_k=8,
                   steps=10, seed=4, goal="hit", target=target, start=start.tokens)  # fmt: skip
    assert abs(end.strength - target) <= abs(start.strength - target) + 1e-5


def test_bad_goals_are_refused(qwen_runner):  # noqa: F811
    import pytest

    for kwargs in ({"goal": "up"}, {"goal": "hit"}, {"goal": "max", "target": 1.0},
                   {"start": [1, 2], "frozen": (0, 1)}):  # fmt: skip
        with pytest.raises(ValueError):
            dream(qwen_runner, 1, 5, "positive", lambdas=(0.0,), steps=1, **kwargs)
