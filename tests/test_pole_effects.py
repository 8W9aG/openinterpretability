"""Pole effects by attribution patching, and the planner's effect-weighted draws (#3486)."""

from __future__ import annotations

import json
import random

import torch

from openinterp.audit_options import selection
from openinterp.coverage import mean_loss
from openinterp.planner_effects import effect_draw, table_path, weighted_pole
from openinterp.pole_effects import estimate, exact_effect, top_poles
from tests.conftest import N_LAYERS, VOCAB


def _ids(seed: int) -> torch.Tensor:
    return torch.randint(1, VOCAB, (8, 12), generator=torch.Generator().manual_seed(seed))


def test_the_estimate_ranks_poles_like_exact_ablation(toy_runner):
    fit, test = _ids(0), _ids(1)
    effects = estimate(toy_runner, fit, test, range(N_LAYERS), 4)
    assert set(effects) == {(la, p) for la in range(N_LAYERS) for p in ("positive", "negative")}
    clean = mean_loss(toy_runner, test, 4)
    poles = [(r["layer"], r["neuron"], r["pole"]) for r in top_poles(effects, 6)]
    poles += [(0, 3, "negative"), (2, 5, "positive"), (1, 7, "negative")]
    est = torch.tensor([float(effects[(la, p)][i]) for la, i, p in poles])
    exact = torch.tensor(
        [exact_effect(toy_runner, fit, test, la, i, p, clean, 4) for la, i, p in poles]
    )
    assert torch.corrcoef(torch.stack([est, exact]))[0, 1] > 0.9


def test_top_poles_are_sorted_largest_first():
    effects = {(0, "positive"): torch.tensor([0.1, 0.3]), (1, "negative"): torch.tensor([0.2])}
    rows = top_poles(effects, 2)
    assert [(r["layer"], r["neuron"], r["pole"]) for r in rows] == [
        (0, 1, "positive"),
        (1, 0, "negative"),
    ]


def _table(tmp_path, poles):
    model = "toy/model"
    path = table_path(tmp_path, model)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"poles": poles}))
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.openinterp]\neffect_share = 1.0\n")
    return model, pyproject


def test_an_effect_draw_names_its_pole_and_selection(tmp_path):
    model, pyproject = _table(
        tmp_path, [{"layer": 2, "neuron": 9, "pole": "negative", "effect": 0.5}]
    )
    c = effect_draw(random.Random(0), model, pyproject, set(), 128)
    assert c.kind == "explore" and c.slug.endswith("L2/N9-negative") and c.layer == 2
    assert "--selection attribution" in c.commands[0] and "--window-tokens 128" in c.commands[0]
    assert "--random" not in c.commands[0]


def test_taken_and_harmless_poles_are_never_drawn(tmp_path):
    rows = [
        {"layer": 0, "neuron": n, "pole": "positive", "effect": e}
        for n, e in ((0, 0.9), (1, -0.2), (2, 0.1))
    ]
    model, _ = _table(tmp_path, rows)
    taken = {"toy--model-L0-N0-positive"}  # a claim key, as open PRs give it
    drawn = {weighted_pole(random.Random(s), model, rows, taken)["neuron"] for s in range(50)}
    assert drawn == {2}


def test_no_table_or_a_zero_share_draws_uniformly(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.openinterp]\neffect_share = 1.0\n")
    assert effect_draw(random.Random(0), "no/table", pyproject, set(), None) is None
    model, pyproject = _table(
        tmp_path, [{"layer": 0, "neuron": 0, "pole": "positive", "effect": 1}]
    )
    pyproject.write_text("[tool.openinterp]\neffect_share = 0.0\n")
    assert effect_draw(random.Random(0), model, pyproject, set(), None) is None


def test_selection_is_random_only_for_random_draws():
    assert selection(True, "attribution") == "random"
    assert selection(False, None) == "manual"
    assert selection(False, "attribution") == "attribution"
