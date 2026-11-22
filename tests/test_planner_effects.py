"""Effect-weighted draws fall back to a uniform draw when no pole is free."""

import json
import random

from openinterpretability.planner_effects import effect_draw, table_path, unit_slug, weighted_pole

MODEL = "toy/model"
ROWS = [
    {"layer": 0, "neuron": 0, "pole": "positive", "effect": 0.9},
    {"layer": 0, "neuron": 1, "pole": "positive", "effect": -0.2},
]


def test_no_free_pole_gives_no_weighted_row():
    taken = {unit_slug(MODEL, ROWS[0])}
    assert weighted_pole(random.Random(0), MODEL, ROWS, taken) is None


def test_a_table_with_every_pole_taken_draws_uniformly(tmp_path):
    path = table_path(tmp_path, MODEL)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"poles": ROWS}))
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.openinterpretability]\neffect_share = 1.0\n")
    taken = {unit_slug(MODEL, ROWS[0])}
    assert effect_draw(random.Random(0), MODEL, pyproject, taken, None) is None
    assert effect_draw(random.Random(0), MODEL, pyproject, set(), None).slug == next(iter(taken))
