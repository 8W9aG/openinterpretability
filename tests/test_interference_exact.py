"""The interference statistics and sides give exact values at their edges (#6491)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from openinterpretability.interference import behaviour, fisher, floors, pruning, report, routes
from openinterpretability.interference.run import _logit_cosine
from openinterpretability.interference.template import pole_mean
from tests.test_interference import _runner


def test_auc_says_why_it_refuses_and_needs_one_label_per_score():
    with pytest.raises(ValueError, match="^auc needs both labels$"):
        fisher.auc([1, 2], [False, False])
    with pytest.raises(ValueError, match="zip"):
        fisher.auc([0.1, 0.2], [True, False, True])


def test_partial_spearman_keeps_a_small_but_real_residual():
    # Each residual has norm about 0.61, so their product is below 1 and above the floor.
    assert fisher.partial_spearman([0, 0, 1], [0, 0, 1], [0, 1, 2]) == pytest.approx(1.0)


def test_the_stratified_null_is_seeded():
    outcomes, strata = [1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]
    first = fisher.stratified_null(sum, outcomes, strata, n=20, seed=3)
    again = fisher.stratified_null(lambda ys: ys[0] + 2 * ys[3], outcomes, strata, n=20, seed=3)
    other = fisher.stratified_null(lambda ys: ys[0] + 2 * ys[3], outcomes, strata, n=20, seed=4)
    assert first["null"] == [3] * 20  # a shuffle within strata keeps each stratum's sum
    assert again == fisher.stratified_null(lambda ys: ys[0] + 2 * ys[3], outcomes, strata,
                                           n=20, seed=3)  # fmt: skip
    assert again["null"] != other["null"]


def test_the_logit_cosine_is_rounded_to_six_places():
    eye = torch.eye(2)
    assert _logit_cosine(eye, eye, torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])) == 0.707107


def test_the_flips_leave_out_the_base_side():
    readings = {"base": {"passes": True}, "changed": {"passes": False},
                "floor:self": {"passes": True}}  # fmt: skip
    assert report._flips(readings) == {"flips": {"changed": True, "floor:self": False},
                                       "counts": True}  # fmt: skip


@pytest.mark.parametrize("value", [torch.tensor(0.5), torch.tensor([[0.0, 0.5, 0.6]])])
def test_a_gate_at_one_half_is_shut(value):
    mask = behaviour._open_mask(value, torch.Size([1, 2]))
    assert mask.tolist() == ([[False, False]] if value.dim() == 0 else [[False, True]])


def test_magnitude_pruning_breaks_ties_by_position():
    assert (
        pruning.zeroed(torch.ones(1000), 0.5, "magnitude").tolist() == [True] * 500 + [False] * 500
    )


def test_pole_mean_signs_and_clamps_the_activations():
    acts = torch.tensor([0.5, -0.25, 0.0, -1.0])
    assert pole_mean(acts, "positive") == pytest.approx(0.125)
    assert pole_mean(acts, "negative") == pytest.approx(0.3125)


def test_a_perturbed_side_draws_seeded_noise_as_large_as_the_change():
    base, other = _runner(0), _runner(1)
    for runner in (base, other):
        runner.model.to(torch.bfloat16)
    sides = [floors.Perturbed(base, other, "random") for _ in range(2)]
    for side in sides:
        side._prepare()
    theirs = dict(other.model.named_parameters())
    for name, weight in base.model.named_parameters():
        noise = sides[0]._noise[name]
        assert noise.dtype == torch.bfloat16 and torch.equal(noise, sides[1]._noise[name])
        change = (theirs[name].detach().float() - weight.detach().float()).norm()
        assert float(noise.float().norm()) == pytest.approx(float(change), rel=0.02)


class _Side:
    def __init__(self, name):
        self.name = name

    @contextmanager
    def entered(self):
        yield self.name


def _routes(monkeypatch, coverages):
    """Rows from explanation_routes, with each side's coverage given and its calls kept."""
    from openinterpretability import explain_coverage
    from openinterpretability.core import answer_records

    calls = []

    def prompt_coverage(runner, prompt, mean, token, at, order, draws):
        calls.append((runner, prompt, mean, token, at, order, draws))
        return {"coverage": coverages[runner]}

    monkeypatch.setattr(explain_coverage, "prompt_coverage", prompt_coverage)
    monkeypatch.setattr(answer_records, "parse_unit", lambda label: f"unit {label}")
    record = SimpleNamespace(prompt="p", position=4,
                             inputs=SimpleNamespace(explained=["a", "b"], token_id=9))  # fmt: skip
    sides = {name: _Side(name) for name in coverages}
    rows = routes.explanation_routes(sides, [record], {name: f"m{name}" for name in coverages})
    return rows, calls


def test_explanation_routes_score_every_explained_unit_as_verified(monkeypatch):
    rows, calls = _routes(monkeypatch, {"base": 0.5, "changed": 0.2, "floor:self": 0.4})
    order = [("unit a", 0.0, "VERIFIED", True), ("unit b", 0.0, "VERIFIED", True)]
    assert calls[0] == ("base", "p", "mbase", 9, 4, order, 0)
    (row,) = rows
    assert row == {"table": "route", "kind": "explanation", "prompt": "p",
                   "coverage": {"base": 0.5, "changed": 0.2, "floor:self": 0.4},
                   "drop": pytest.approx({"changed": 0.3, "floor:self": 0.1}),
                   "counts": True}  # fmt: skip


@pytest.mark.parametrize(("coverages", "counts"), [
    ({"base": routes.TOL, "changed": 0.0}, False),  # no floor: the drop must pass TOL
    ({"base": 0.3, "changed": 0.2, "floor:self": 0.5}, True),  # a floor that gains is 0
    ({"base": 0.3, "changed": 0.2, "floor:self": 0.25}, True),
    ({"base": 0.3, "changed": 0.2, "floor:self": 0.15}, False),  # the floor drops further
    ({"base": None, "changed": None}, False),  # no coverage counts as 0
    ({"base": 0.5, "changed": None}, True),
])  # fmt: skip
def test_explanation_routes_count_a_drop_only_above_every_floor(monkeypatch, coverages, counts):
    rows, _ = _routes(monkeypatch, coverages)
    assert rows[0]["counts"] is counts
