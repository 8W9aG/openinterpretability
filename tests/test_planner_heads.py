"""Explore draws of attention heads (openinterp/planner_heads.py, issue #3248)."""

import random
from types import SimpleNamespace

import pytest

from openinterp import planner, planner_heads
from openinterp.draw_steps import draw_slug, random_head
from openinterp.evidence import ScanRequest, pick_unit

MODEL = "Qwen/Qwen2.5-0.5B"


def config(tmp_path, body: str):
    path = tmp_path / "pyproject.toml"
    path.write_text(f"[tool.openinterp]\n{body}\n")
    return path


def test_the_share_is_honoured_over_many_seeds():
    heads = sum(planner_heads.is_head_draw(seed, 0.1) for seed in range(20_000))
    assert 0.09 < heads / 20_000 < 0.11
    assert not any(planner_heads.is_head_draw(seed, 0.0) for seed in range(1_000))


def test_head_share_reads_the_config_or_the_default(tmp_path):
    assert planner_heads.head_share(config(tmp_path, "head_share = 0.25")) == 0.25
    assert planner_heads.head_share(tmp_path / "none.toml") == planner_heads.DEFAULT_HEAD_SHARE


def test_a_head_draw_audits_a_random_head_and_skips_the_effect_draw(tmp_path, monkeypatch):
    def no_effect_draw(*args, **kwargs):
        raise AssertionError("a head draw must not reach the effect draw")

    monkeypatch.setattr("openinterp.planner_effects.effect_draw", no_effect_draw)
    pyproject = config(tmp_path, "head_share = 1.0")
    rng = random.Random(0)
    got = planner._explore(rng, [MODEL], {MODEL: 1.0}, pyproject, MODEL)
    assert got.kind == "explore" and "--unit head" in got.commands[0]
    assert "--random" in got.audit and got.skills == ["attention-head-audit"]


def test_with_no_head_share_the_draw_is_a_neuron_pole(tmp_path):
    pyproject = config(tmp_path, "head_share = 0.0")
    got = planner._explore(random.Random(0), [MODEL], {MODEL: 1.0}, pyproject, MODEL)
    assert "--unit" not in got.commands[0]


@pytest.mark.parametrize("seed", [0, 1, 953380901, 3280387012])
def test_random_head_matches_the_audits_draw(seed):
    runner = SimpleNamespace(n_layers=24, n_heads=14)
    request = ScanRequest(model_name=MODEL, random_pick=True, seed=seed, unit="head")
    ident, _ = pick_unit(runner, request, None)
    assert random_head(MODEL, seed, (24, 14)) == ident.slug


def test_the_claim_names_the_head_before_the_scan(monkeypatch):
    monkeypatch.setattr("openinterp.draw_steps.model_heads", lambda model: (24, 14))
    audit = f"--model {MODEL} --random --seed 7 --unit head"
    choice = SimpleNamespace(slug=None, audit=audit, model=MODEL)
    assert draw_slug(choice) == random_head(MODEL, 7, (24, 14))
    assert "/H" in draw_slug(choice)
