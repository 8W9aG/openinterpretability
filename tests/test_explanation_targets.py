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
                           record_path=path, standing=(status, None))  # fmt: skip


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


def test_the_run_stops_when_the_share_reaches_the_target(registry, tmp_path, capsys):
    # The share is measured before each draw, after the previous experiment landed (#4473).
    shares = iter([0.5, 0.82])
    draws = TargetedDraws([A, B], reviewer="rev/model", meter=lambda: next(shares), target=0.8)
    assert draws(tmp_path, claims=[]).choice.slug == A.slug
    with pytest.raises(Finished, match="the measure 82.0% reached the target 80%"):
        draws(tmp_path, claims=[])
    out = capsys.readouterr().out
    assert "measured 50.0%, target 80%" in out and "measured 82.0%" in out


def test_run_n_stops_after_n_experiments(registry, tmp_path):
    draws = TargetedDraws([A, B], reviewer="rev/model", limit=1)
    draws(tmp_path, claims=[])
    with pytest.raises(Finished, match="ran 1 experiments"):
        draws(tmp_path, claims=[])


def test_an_unstudied_superposition_gets_its_draw(registry, tmp_path, monkeypatch):
    from openinterpretability import planner_superpositions as ps

    seen = {}

    def candidate(ident, definition, source, window):
        seen.update(definition=definition, window=window)
        return _candidate("superposition", 1.0)

    monkeypatch.setattr(ps, "_candidate", candidate)
    monkeypatch.setattr(ps, "definition_identifier", lambda d: "ident")
    s = Target("m/L2/S0123456789ab", "m/x", 2, "", "L2/S0123456789ab", {"members": []})
    draws = TargetedDraws([s], reviewer="rev/model", window=64)
    assert draws(tmp_path, claims=[]).choice.kind == "superposition"
    assert seen == {"definition": {"members": []}, "window": 64}


def test_targets_order_and_circuit_notes():
    from openinterpretability.explain_groups import GroupDef
    from openinterpretability.explain_plan import PlannedExperiment
    from openinterpretability.explanation_targets import by_gain, circuit_notes, targets

    def planned(target, entity, gain, units=()):
        commands = ["open-interpretability audit x"]
        return PlannedExperiment(target, entity, 0.1, "unstudied", "explore", "why", commands,
                                 units=list(units), gain=gain)  # fmt: skip

    unit = planned("L1/N2+", "pole", 0.01, [(1, 2, "positive")])
    sup = planned("L2/S0123456789ab", "superposition", 0.05)
    circ = planned("C4", "circuit", 0.2)
    ordered = by_gain([unit, sup, circ])
    assert [e.target for e in ordered] == ["C4", "L2/S0123456789ab", "L1/N2+"]
    group = GroupDef("superposition", "L2/S0123456789ab", "m--x/L2/S0123456789ab", ((2, 1, 1.0),),
                     definition={"members": []})  # fmt: skip
    found = targets(ordered, "m/x", [group])
    assert [t.label for t in found] == ["L2/S0123456789ab", "L1/N2+"]  # the circuit is not run
    assert found[0].definition == {"members": []} and found[0].layer == 2
    (note,) = circuit_notes(ordered)
    assert note.startswith("C4 (10.0%): its task needs a corrupted twin of this prompt")


def test_the_meter_counts_verified_units_and_units_in_verified_groups(tmp_path, monkeypatch):
    from openinterpretability import explain, explain_groups
    from openinterpretability.explain_groups import GroupDef
    from openinterpretability.explanation_targets import ShareMeter

    slug = explain._slug("m/x", (1, 2, "positive"))
    labels = {slug: ("VERIFIED", None, None)}
    monkeypatch.setattr(explain, "registry_labels", lambda wiki, model: labels)
    circuits = {"C4": ("VERIFIED", None)}
    monkeypatch.setattr(explain_groups, "circuit_labels", lambda wiki, model: circuits)
    order = [((1, 2, "positive"), 3.0, "", False), ((5, 1, "head"), 1.0, "", False),
             ((6, 7, "negative"), 4.0, "", False)]  # fmt: skip
    circuit = GroupDef("circuit", "C4", "C4", (), heads=((5, 1),))
    meter = ShareMeter("m/x", order, [circuit], tmp_path)
    head_key = circuit.units[0]
    order[1] = (head_key, 1.0, "", False)
    assert meter() == pytest.approx(4.0 / 8.0)  # the VERIFIED pole, and the head in VERIFIED C4


def test_a_target_already_done_is_skipped_on_later_draws(registry, tmp_path):
    registry.stops = {A.slug}
    draws = TargetedDraws([A, B], reviewer="rev/model")
    assert draws(tmp_path, claims=[]).choice.slug == B.slug
    assert draws.done == {A.slug: "stopped (dead or default-on)"}
    registry.stops = set()  # A stays done: it is not looked at again
    assert draws(tmp_path, claims=[]).choice.slug == B.slug
    assert draws.attempts == {B.slug: 2}


def test_the_meter_counts_the_poles_of_a_verified_superposition(tmp_path, monkeypatch):
    from openinterpretability import explain, explain_groups
    from openinterpretability.explain_groups import GroupDef
    from openinterpretability.explanation_targets import ShareMeter

    slug = "m--x/L2/S0123456789ab"
    monkeypatch.setattr(explain, "registry_labels", lambda wiki, model: {slug: ("VERIFIED",)})
    monkeypatch.setattr(explain_groups, "circuit_labels", lambda wiki, model: {})
    sup = GroupDef("superposition", "L2/S0123456789ab", slug, ((2, 1, 1.0),))
    order = [((2, 1, "positive"), 3.0, "", False), ((6, 7, "negative"), 1.0, "", False)]
    assert ShareMeter("m/x", order, [sup], tmp_path)() == pytest.approx(3.0 / 4.0)


def test_each_draw_counts_and_reads_the_wiki_it_is_given(registry, tmp_path, monkeypatch):  # #6066
    asked, slugs = [], []
    monkeypatch.setattr(planner, "stopped", lambda wiki: asked.append(wiki) or [])
    monkeypatch.setattr(
        planner, "candidates_for", lambda state, s, *a: slugs.append(s) or iter(registry.follow)
    )
    registry.states = {A.slug: _state("REJECTED")}
    registry.follow = [_candidate("revise", 0.9)]
    draws = TargetedDraws([A, B], reviewer="rev/model")
    plan = draws(tmp_path, claims=[])
    draws(tmp_path, claims=[])
    assert draws.started == 2 and asked == [tmp_path, tmp_path] and slugs[0] == {A.slug}
    assert plan.explored is False and plan.seed == 0
