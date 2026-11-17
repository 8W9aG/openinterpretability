"""Draws that run an answer's planned experiments until each unit is VERIFIED (#3744)."""

from types import SimpleNamespace

import pytest

from openinterpretability import experiment_loop, planner
from openinterpretability.experiment_loop import Finished
from openinterpretability.explanation_targets import Target, TargetedDraws

A = Target("m/L1/N2-positive", "m/x", 1, "open-interpretability audit --model m/x --layer 1 "
           "--neuron 2 --pole positive --selection attribution", "L1/N2+")  # fmt: skip
B = Target("m/L3/H4", "m/x", 3, "open-interpretability audit --model m/x --layer 3 --head 4 "
           "--selection attribution", "L3/H4")  # fmt: skip


def _state(status, path="wiki/x.json"):
    return SimpleNamespace(record=SimpleNamespace(verdict=SimpleNamespace(status=status)),
                           record_path=path)  # fmt: skip


@pytest.fixture
def registry(monkeypatch):
    """The registry's states, stops and follow-ups, which each test sets."""
    world = SimpleNamespace(states={}, stops=set(), follow=[], proposer="other/model")
    monkeypatch.setattr(planner, "load_states", lambda wiki: dict(world.states))
    stops = lambda wiki: [SimpleNamespace(slug=s) for s in world.stops]  # noqa: E731
    monkeypatch.setattr(planner, "stopped", stops)
    monkeypatch.setattr(planner, "candidates_for", lambda *a: iter(world.follow))
    monkeypatch.setattr(experiment_loop, "proposer_of", lambda record: world.proposer)
    return world


def _candidate(kind, weight, record="wiki/x.json"):
    return planner.Candidate(kind=kind, model="m/x", weight=weight, rationale=kind,
                             commands=[f"run {kind}"], slug=A.slug, record=record)  # fmt: skip


def test_an_unstudied_target_gets_its_attribution_audit(registry, tmp_path):
    draws = TargetedDraws([A, B], reviewer="rev/model")
    plan = draws(tmp_path, claims=[])
    assert plan.choice.kind == "explore" and plan.choice.slug == A.slug
    assert plan.choice.commands == [A.audit] and "--selection attribution" in plan.choice.audit


def test_a_studied_target_gets_its_heaviest_follow_up_until_verified(registry, tmp_path):
    registry.states = {A.slug: _state("REJECTED")}
    registry.follow = [_candidate("reevaluate", 0.2), _candidate("revise", 0.9)]
    draws = TargetedDraws([A], reviewer="rev/model")
    assert draws(tmp_path, claims=[]).choice.kind == "revise"
    registry.states = {A.slug: _state("NEEDS_REVISION")}  # the revise ran, not VERIFIED yet
    assert draws(tmp_path, claims=[]).choice.kind == "revise"  # so it runs again
    registry.states = {A.slug: _state("VERIFIED")}
    with pytest.raises(Finished, match="L1/N2\\+ VERIFIED"):
        draws(tmp_path, claims=[])
    assert draws.attempts == {A.slug: 2}


def test_self_reviews_stopped_claimed_and_exhausted_targets_are_passed_over(registry, tmp_path):
    registry.states = {A.slug: _state("NEEDS_REVISION")}
    registry.follow = [_candidate("independent_review", 0.9)]
    registry.proposer = "rev/model"  # the reviewer wrote the record: it can't review it
    registry.stops = {B.slug}
    with pytest.raises(Finished) as done:
        TargetedDraws([A, B], reviewer="rev/model")(tmp_path, claims=[])
    assert "no follow-up the reviewer can run" in str(done.value) and "stopped" in str(done.value)
    registry.proposer, registry.stops = "other/model", set()
    claimed = TargetedDraws([B], reviewer="rev/model")
    with pytest.raises(Finished, match="waiting on an open PR"):
        claimed(tmp_path, claims=[planner.claim_key(B.slug)])
    tired = TargetedDraws([B], reviewer="rev/model", max_attempts=1)
    tired(tmp_path, claims=[])
    with pytest.raises(Finished, match="not VERIFIED after 1 runs"):
        tired(tmp_path, claims=[])


def test_finished_ends_the_loop_cleanly(tmp_path, monkeypatch):
    def finished(*a, **k):
        raise Finished("all done")

    monkeypatch.setattr(experiment_loop, "iteration", finished)
    loop = experiment_loop.Loop(experiment_loop.Options("p", "r"), tmp_path, 0, 3, None, None,
                                lambda s: None)  # fmt: skip
    assert loop.run() == 0
