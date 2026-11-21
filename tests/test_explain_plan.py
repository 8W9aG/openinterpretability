"""The experiments a complete explanation needs (explain_plan.py, #3742)."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterpretability import explain as ex
from openinterpretability import explain_plan as ep
from openinterpretability.explain_groups import HEAD, GroupDef
from tests.test_explain import PROMPT, TEXTS, means  # noqa: F401  (a fixture)


def _slug(unit):
    return ex._slug("toy", unit)


def test_the_plan_follows_attribution_and_stops_at_the_target(toy_runner, means):  # noqa: F811
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=3, edges=0)
    full, half = (ep.plan(result, {}, "toy", target=t) for t in (1.0, 0.5))
    shares = [e.share for e in full.experiments]
    assert shares == sorted(shares, reverse=True) and full.reaches == pytest.approx(1.0)
    assert half.reaches >= 0.5 and half.reaches - half.experiments[-1].share < 0.5
    assert [e.target for e in half.experiments] == [e.target for e in full.experiments][
        : len(half.experiments)
    ]


def test_explained_units_are_skipped_and_stopped_ones_are_blocked(toy_runner, means):  # noqa: F811
    first = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=2, edges=0)
    top, second = first.poles
    labels = {_slug(top.key): ("VERIFIED", "x", 1.0), _slug(second.key): ("stopped", None, None)}
    result = ex.explain(toy_runner, "toy", PROMPT, means, labels, top=2, edges=0)
    planned = ep.plan(result, {}, "toy", target=1.0)
    targets = [e.target for e in planned.experiments]
    assert top.label not in targets and second.label not in targets
    assert [b["target"] for b in planned.blocked_units] == [second.label]
    assert planned.blocked == pytest.approx(abs(second.attribution) / result.total)
    assert planned.planned_share == pytest.approx(1 - result.verified_share - planned.blocked)


def test_an_unstudied_unit_gets_an_audit_chosen_by_attribution():
    pole = ep.audit_command("m/x", (3, 7, "negative"), 128)
    assert pole == (
        "open-interpretability audit --model m/x --layer 3 --neuron 7 --pole negative "
        "--selection attribution --window-tokens 128"
    )
    head = ep.audit_command("m/x", (2, 4, HEAD), None)
    assert (
        head == "open-interpretability audit --model m/x --layer 2 --head 4 --selection attribution"
    )


def test_a_studied_unit_gets_the_planners_follow_up(toy_runner, means, monkeypatch):  # noqa: F811
    first = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=1, edges=0)
    (top,) = first.poles
    labels = {_slug(top.key): ("REJECTED", None, 1.0)}
    result = ex.explain(toy_runner, "toy", PROMPT, means, labels, top=1, edges=0)
    follow = SimpleNamespace(kind="revise", rationale="revise it", commands=["c1", "c2"], weight=1)
    monkeypatch.setattr(ep, "_follow_up", lambda state, slugs, now: follow)
    planned = ep.plan(result, {_slug(top.key): object()}, "toy")
    (e,) = [e for e in planned.experiments if e.target == top.label]
    assert (e.status, e.draw, e.commands) == ("REJECTED", "revise", ["c1", "c2"])
    monkeypatch.setattr(ep, "_follow_up", lambda state, slugs, now: None)
    blocked = ep.plan(result, {_slug(top.key): object()}, "toy")
    assert "no follow-up" in blocked.blocked_units[0]["reason"]


def test_unverified_circuits_get_a_task_hypothesis(toy_runner, means):  # noqa: F811
    circuit = GroupDef("circuit", "C1", "C1", ((0, 1, 1.0), (1, 2, 1.0)))
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, groups=[circuit], edges=0)
    (planned,) = ep.plan(result, {}, "toy", groups=[circuit]).groups
    assert planned.entity == "circuit" and planned.draw == "circuit"
    assert "circuits propose --model toy --circuit C1" in planned.commands[0]
    verified = {"C1": ("VERIFIED", "x")}
    done = ex.explain(toy_runner, "toy", PROMPT, means, {}, groups=[circuit],
                      group_labels=verified, edges=0)  # fmt: skip
    assert ep.plan(done, {}, "toy", groups=[circuit]).groups == []
    full = ep.plan(result, {}, "toy", groups=[circuit])
    held = [e for e in full.experiments if e.covered_by]
    assert held and all(e.covered_by == ["C1"] for e in held)


def test_cli_lists_the_plan(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(tmp_path / "wiki"), "--experiments", "1", "--edges", "0"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    assert (
        out.output.count("$ open-interpretability audit") == 1
        and "more unit experiments" in out.output
    )
    as_json = json.loads(CliRunner().invoke(cli, [*args, "--json", "--target", "0.5"]).output)
    plan = as_json["coverage_plan"]  # --target is causal coverage (#4476)
    assert plan["target"] == 0.5 and plan["reach"] >= plan["now"] and "order" not in as_json
    assert plan["experiments"] is None or 0 <= plan["experiments"] <= as_json["planned"]
    assert len(as_json["experiments"]) == 1 <= as_json["planned"]  # only the next --experiments
    assert as_json["coverage"]["coverage"] == 0.0  # nothing is explained yet


def test_prompt_coverage_is_what_the_explained_units_bring_back(toy_runner, means):  # noqa: F811
    from openinterpretability.explain_coverage import prompt_coverage

    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=1000, edges=0)
    token = result.token_id
    none = prompt_coverage(toy_runner, PROMPT, means, token, -1, result.order)
    assert none["coverage"] == 0.0 and none["units"] == 0 and none["random"] is None
    everything = [(u, s, st, True) for u, s, st, _ in result.order]
    full = prompt_coverage(toy_runner, PROMPT, means, token, -1, everything)
    assert full["coverage"] == pytest.approx(1.0, abs=1e-4)  # every active unit kept: clean
    some = [(u, s, st, i < 5) for i, (u, s, st, _) in enumerate(result.order)]
    top5 = prompt_coverage(toy_runner, PROMPT, means, token, -1, some, draws=3)
    assert top5["units"] == 5 and top5["random"] is not None


def test_cli_run_hands_the_top_experiments_to_the_loop(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    seen = {}

    def run_loop(proposer, reviewer, plan_next, out=None):
        seen.update(proposer=proposer, reviewer=reviewer, draws=plan_next)
        return 0

    monkeypatch.setattr("openinterpretability.cli_loop.run_loop", run_loop)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(tmp_path / "wiki"), "--edges", "0", "--run", "2",
            "--proposer", "anthropic/p", "--reviewer", "anthropic/r",
            "--max-attempts", "3"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    draws = seen["draws"]
    assert (seen["proposer"], seen["reviewer"], draws.max_attempts) == (
        "anthropic/p",
        "anthropic/r",
        3,
    )
    assert draws.limit == 2 and len(draws.targets) >= 2  # --run 2 stops after 2 (#4473)
    assert "experiments, by coverage gain, until the coverage reaches" in out.output
    assert all("--selection attribution" in t.audit for t in draws.targets)
    external = CliRunner().invoke(
        cli, [*args[:-4], "--proposer", "external/x", "--reviewer", "anthropic/r"]
    )
    assert external.exit_code != 0  # nobody would answer an external model's prompt files


def test_cli_records_the_explanation(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli
    from openinterpretability.core.explanation_records import load_all

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    wiki = tmp_path / "wiki"
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(wiki), "--edges", "0", "--record", "--no-land"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    (((path, record),),) = load_all(wiki).values()
    assert f"Explanation → {path}" in out.output
    assert record.model_name == "toy" and record.prompt == PROMPT and record.narration is None
    assert record.units and record.planned_experiments > 0


def test_an_audit_command_takes_only_a_known_pole():
    with pytest.raises(ValueError):
        ep.audit_command("m/x", (1, 2, "positive;x"), None)


@pytest.mark.parametrize("code", [0, 1])
def test_run_then_record_the_final_state(toy_runner, tmp_path, monkeypatch, code):
    # The record is written after the loop, from the registry as the run left it (#4474).
    from openinterpretability import cli_explain, experiment_loop
    from openinterpretability.cli import cli
    from openinterpretability.core.explanation_records import load_all

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    runner = lambda *a, **k: toy_runner  # noqa: E731
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", runner)
    monkeypatch.setattr(experiment_loop, "reset", lambda: None)
    analyses = []
    real = cli_explain._analyse
    monkeypatch.setattr(cli_explain, "_analyse", lambda *a: analyses.append(1) or real(*a))
    landed = experiment_loop.Result("toy/L1/N2-positive", "VERIFIED", "https://x/pull/1", True)

    def run_loop(proposer, reviewer, plan_next, out=None):
        out["loop"] = SimpleNamespace(results=[landed], finished=None if code else "reached")
        return code

    monkeypatch.setattr("openinterpretability.cli_loop.run_loop", run_loop)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    wiki = tmp_path / "wiki"
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(wiki), "--edges", "0", "--run", "--record", "--no-land",
            "--level", "mechanism", "--proposer", "anthropic/p",
            "--reviewer", "anthropic/r"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == code, out.output
    assert len(analyses) == 2  # before the run, to plan it, and after it
    (((_, record),),) = load_all(wiki).values()
    assert record.run.landed == ["toy/L1/N2-positive VERIFIED https://x/pull/1"]
    assert record.run.share_after == record.explained_share
    failed = "the loop stopped with exit code 1: failures or a rate limit"
    assert record.run.stopped == (failed if code else "reached")


def test_complete_runs_narrates_and_lands_in_one_command(toy_runner, tmp_path, monkeypatch):
    # --complete is --run to the target, --level long and --record, landed (#4476).
    from openinterpretability import cli_explain, experiment_loop, explain_land
    from openinterpretability.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    runner = lambda *a, **k: toy_runner  # noqa: E731
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", runner)
    monkeypatch.setattr(experiment_loop, "reset", lambda: None)
    seen = {}

    def run_loop(proposer, reviewer, plan_next, out=None):
        seen["limit"] = plan_next.limit
        out["loop"] = SimpleNamespace(results=[], finished="no target left")
        return 0

    narrative = {"sections": [{"title": "Step", "text": "Mostly unknown."}],
                 "paragraph": "Mostly unknown."}  # fmt: skip
    told = {"narrative": narrative, "narrator_model": "external/x",
            "citations_not_in_evidence": []}  # fmt: skip
    monkeypatch.setattr("openinterpretability.cli_loop.run_loop", run_loop)
    monkeypatch.setattr(cli_explain, "_narrate", lambda *a: told)
    landed = []

    def land_record(record):
        landed.append(record)
        return "https://x/pull/9", True

    monkeypatch.setattr(explain_land, "land_record", land_record)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(tmp_path / "wiki"), "--edges", "0", "--complete", "--target", "0.8",
            "--proposer", "anthropic/p", "--reviewer", "anthropic/r"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    assert seen["limit"] == 0  # to the target, with no cap
    (record,) = landed
    assert record.narration.paragraph == "Mostly unknown."
    assert record.run.stopped == "no target left"
    assert "Explanation → https://x/pull/9 (merged)" in out.output
    assert out.output.strip().splitlines()[-1].startswith("Reached: prompt coverage ")


def test_the_reached_line():
    from openinterpretability.cli_explain import reached

    ran = SimpleNamespace(share_before=0.1)
    line = reached(0.42, 0.8, {"coverage": 0.3, "random": 0.05}, ran)
    assert line == (
        "Reached: prompt coverage 30.0% (from 10.0% before the run), target 80%, "
        "against 5.0% for random units matched by kind and layer; explained share of "
        "|attribution| 42.0%."
    )
    assert reached(0.0, 1.0, {"coverage": None}) == (
        "Reached: no prompt coverage, target 100%; explained share of |attribution| 0.0%."
    )


def test_coverage_plan_finds_the_fewest_experiments_for_the_target(toy_runner, means):  # noqa: F811
    from openinterpretability.explain_coverage import coverage_plan

    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=5, edges=0)
    planned = ep.plan(result, {}, "toy", target=1.0)
    args = (toy_runner, PROMPT, means, result.token_id, -1, result.order)
    now = coverage_plan(*args, planned, 0.5)["now"]
    assert coverage_plan(*args, planned, now)["experiments"] == 0  # reached already
    found = coverage_plan(*args, planned, 0.5)
    if found["experiments"] is not None:  # the smallest k whose units bring back half
        assert 0 < found["experiments"] <= len(planned.experiments)
    out_of_reach = coverage_plan(*args, planned, found["reach"] + 0.5)
    assert out_of_reach["experiments"] is None and out_of_reach["reach"] == found["reach"]


def test_the_coverage_meter_counts_only_coverage_above_random(monkeypatch):
    from openinterpretability import explain_coverage, explanation_targets

    monkeypatch.setattr(explanation_targets, "explained_now", lambda *a: {"u"})
    results = iter([{"coverage": 0.3, "random_draws": [0.1, 0.2]},
                    {"coverage": 0.3, "random_draws": [0.1, 0.4]}])  # fmt: skip
    monkeypatch.setattr(explain_coverage, "prompt_coverage", lambda *a, **k: next(results))
    meter = explanation_targets.CoverageMeter(None, None, "p", 1, -1, "m",
                                              [("u", 1.0, "", False)], [], None)  # fmt: skip
    assert meter() == 0.3 and "best random set 20.0%" in meter.note
    assert meter() == 0.0  # a random set did as well, so the coverage isn't credited


def _fake_plan(monkeypatch, coverage_of):
    """A plan of three unit experiments on a fake model whose coverage is ``coverage_of(units)``."""
    from types import SimpleNamespace

    from openinterpretability import explain_coverage as ec

    monkeypatch.setattr(ec, "keep_hooks", lambda runner, unit_means, units: list(units))
    # clean −1, all ablated −5: coverage c means a log-probability of −5 + 4c
    monkeypatch.setattr(ec, "_log_prob", lambda r, p, t, pos, hooks=None: (
        -1.0 if hooks is None else -5.0 + 4.0 * coverage_of(set(hooks))))  # fmt: skip
    experiments = [ep.PlannedExperiment(target=name, entity="pole", share=0.1, status="unstudied",
                                        draw="explore", rationale="", commands=[], units=[name])
                   for name in ("a", "b", "c")]  # fmt: skip
    return ec, SimpleNamespace(experiments=experiments, groups=[])


def test_the_greedy_plan_picks_by_joint_coverage_not_attribution_order(monkeypatch):
    # "a" (first by attribution) adds little alone; "c" adds most; "b" only helps with "c" (#4558).
    table = {frozenset(): 0.0, frozenset("a"): 0.05, frozenset("b"): 0.0, frozenset("c"): 0.3,
             frozenset("bc"): 0.6, frozenset("ac"): 0.32, frozenset("ab"): 0.05,
             frozenset("abc"): 0.4}  # fmt: skip
    ec, planned = _fake_plan(monkeypatch, lambda units: table[frozenset(units)])
    found = ec.coverage_plan(None, "p", None, 0, -1, [], planned, 0.5)
    assert found["chosen"] == ["c", "b"] and found["experiments"] == 2
    assert [e.target for e in planned.experiments] == ["c", "b", "a"]  # the picks first
    assert planned.experiments[1].gain == pytest.approx(0.3)  # b's marginal gain, with c
    assert found["reach"] == pytest.approx(0.6) and found["now"] == 0.0


def test_the_greedy_plan_stops_when_nothing_helps(monkeypatch):
    ec, planned = _fake_plan(monkeypatch, lambda units: 0.2 if units == {"a"} else 0.0)
    found = ec.coverage_plan(None, "p", None, 0, -1, [], planned, 0.5)
    assert found["chosen"] == ["a"] and found["experiments"] is None  # out of reach
    assert ec.coverage_plan(None, "p", None, 0, -1, [], planned, 0.0)["experiments"] == 0


def _toy_superposition():
    from openinterpretability.core.profiles import SuperpositionProfile
    from openinterpretability.planner_superpositions import definition_identifier

    members = [{"neuron_idx": 1, "weight": 1.0}, {"neuron_idx": 2, "weight": -1.0}]
    profile = SuperpositionProfile.model_validate(
        {"members": members, "method": "c", "source": "t"}
    )
    definition = {"model": "toy", "layer": 0, **profile.model_dump(mode="json")}
    ident = definition_identifier(definition)
    name = ident.slug.split("/", 1)[1]
    group = GroupDef("superposition", name, ident.slug, ((0, 1, 1.0), (0, 2, -1.0)),
                     definition=definition)  # fmt: skip
    return group, ident, profile


def test_an_unstudied_superposition_gets_the_planners_draw(toy_runner, means):  # noqa: F811
    group, ident, _ = _toy_superposition()
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, groups=[group], edges=0)
    (planned,) = ep.plan(result, {}, "toy", groups=[group]).groups
    assert planned.entity == "superposition" and planned.draw == "superposition"
    assert planned.commands[0].startswith(
        f"open-interpretability superpositions show {ident.superposition_id} --model toy"
    )
    assert "an unstudied superposition of 2 neurons on layer 0" in planned.rationale


def test_a_studied_superposition_gets_its_heaviest_follow_up(toy_runner, means, record, tmp_path):  # noqa: F811
    from openinterpretability.core.schemas import AuditRecord
    from openinterpretability.planner_states import NeuronState

    group, ident, profile = _toy_superposition()
    submission = record.submission.model_copy(
        update={"identifier": ident, "superposition_profile": profile}
    )
    verdict = record.verdict.model_copy(update={"status": "NEEDS_REVISION"})
    studied = AuditRecord(submission=submission, verdict=verdict)
    path = tmp_path / "r.json"
    state = NeuronState(ident.slug, studied, path, experiments=1, last=verdict.timestamp)
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, groups=[group], edges=0)
    (planned,) = ep.plan(result, {ident.slug: state}, "toy", groups=[group]).groups
    # The heaviest follow-up on the record, not a new audit of the definition.
    assert planned.draw == "reevaluate"
    assert planned.commands == [f"open-interpretability reevaluate --record {path}"]
