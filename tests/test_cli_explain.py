"""`open-interpretability explain` (openinterpretability/cli_explain.py): options and printing."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterpretability import cli_explain
from openinterpretability import explain as ex
from openinterpretability.cli import cli
from openinterpretability.explain_plan import CAVEAT, Plan, PlannedExperiment
from tests.test_explain import PROMPT, TEXTS


@pytest.fixture
def explain_args(toy_runner, tmp_path, monkeypatch):
    """The arguments of an `explain` of the toy model, with its corpus and an empty registry."""
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr(cli_explain, "_load_runner", lambda *a, **k: toy_runner)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    return ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(tmp_path / "wiki"), "--edges", "0", "--top", "3"]  # fmt: skip


def _invoke(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_the_answer_token_must_be_one_token(explain_args, toy_runner):
    (token,) = toy_runner.tokenizer.encode("e")
    result = _invoke(*explain_args, "--token", "e", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["token_id"] == token
    result = _invoke(*explain_args, "--token", "ef")
    assert result.exit_code != 0 and "'ef' is 2 tokens, not one" in result.output


def test_json_with_evidence_out_also_writes_the_result_there(explain_args, tmp_path):
    out = tmp_path / "fix" / "evidence.json"
    result = _invoke(*explain_args, "--json", "--evidence-out", out)
    assert result.exit_code == 0, result.output
    assert out.read_text() == result.output
    assert json.loads(out.read_text())["coverage"]["coverage"] == 0.0


def test_an_explanation_at_the_target_only_gains_the_new_level(explain_args, tmp_path, monkeypatch):
    from openinterpretability.agents.narrator import Narrative, Narrator
    from openinterpretability.core.explanation_records import CoverageDetail, load_all
    from tests.test_explain_reuse import INPUTS
    from tests.test_explanation_records import _record, _write

    wiki = tmp_path / "wiki"
    old = _record(datetime(2026, 11, 1, tzinfo=UTC)).model_copy(
        update={
            "model_name": "toy",
            "prompt": PROMPT,
            "prompt_coverage": 0.5,
            "narration": None,
            "coverage_detail": CoverageDetail(),
            "inputs": INPUTS,
        }  # fmt: skip
    )
    _write(wiki, old)
    ran = []
    monkeypatch.setattr(cli_explain, "run_plan", lambda *a, **k: ran.append(a))
    story = Narrative(sections=[{"title": "Step", "text": "Not known."}], paragraph="Not known.")
    monkeypatch.setattr(Narrator, "narrate", lambda self, ev: story)
    result = _invoke(*explain_args, "--target", "0.4", "--run", "--level", "paragraph",
                     "--narrator-model", "anthropic/x", "--record", "--no-land")  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "No experiments run; this only adds the paragraph level." in result.output
    assert ran == []  # --run is dropped: the earlier explanation reaches the target
    (records,) = load_all(wiki).values()
    new = records[-1][1]
    assert new.previous == old.id and new.reused and new.run is None
    assert new.narration.paragraph == "Not known."


# -- printing the plan and the coverage ---------------------------------------------------


def _experiment(target, share, gain=None, covered_by=()):
    return PlannedExperiment(
        target=target, entity="pole", share=share, status="unstudied", draw="explore",
        rationale="unstudied", commands=[f"open-interpretability audit {target}"],
        covered_by=list(covered_by), gain=gain,
    )  # fmt: skip


def _plan(experiments, *, explained=0.0, target=1.0, blocked=0.0, blocked_units=()):
    return Plan(target=target, explained=explained, planned_share=sum(e.share for e in experiments),
                blocked=blocked, experiments=experiments, groups=[],
                blocked_units=list(blocked_units))  # fmt: skip


REACH = {"experiments": 2, "target": 1.0, "now": 0.1, "reach": 0.9}


def test_the_plan_names_an_overshoot_and_the_blocked_share(capsys):
    plan = _plan(
        [_experiment("L0/N1+", 0.3, gain=0.05, covered_by=["C4"])],
        blocked=0.2, blocked_units=[{"target": "L1/N2+", "share": 0.2, "reason": "stopped"}],
    )  # fmt: skip
    cli_explain._print_plan(plan, 5, REACH | {"overshoot": True})
    out = capsys.readouterr().out
    assert "with every planned unit kept, the answer overshoots the clean model" in out
    assert "Greedy picks" not in out  # no greedy picks were made
    assert "+5.0% coverage" in out and "; in C4" in out
    assert "more unit experiments" not in out  # every experiment is listed
    assert "Blocked: 20.0% of the attribution, on 1 units" in out
    assert out.rstrip().endswith(CAVEAT.rstrip())


def test_the_plan_lists_no_milestones_when_the_experiments_fall_short(capsys):
    plan = _plan([_experiment("L0/N1+", 0.1)], explained=0.0, target=1.0)
    assert cli_explain._milestones(plan) == []  # 10% never reaches the 50% mark
    cli_explain._print_plan(plan, 5, REACH)
    out = capsys.readouterr().out
    assert "Of the |attribution|" not in out and "gain not measured" in out
    assert "Blocked:" not in out


def test_milestones_count_the_experiments_that_reach_each_mark():
    plan = _plan([_experiment("a", 0.3), _experiment("b", 0.3), _experiment("c", 0.35)])
    assert cli_explain._milestones(plan) == [(0.5, 2), (0.8, 3), (0.9, 3)]


def test_coverage_says_when_ablating_every_unit_changes_nothing(capsys):
    cli_explain._print_coverage({"coverage": None, "random": None, "units": 0})
    assert capsys.readouterr().out == (
        "Prompt coverage: ablating every unit doesn't change the answer.\n"
    )


def test_coverage_names_an_overshoot_and_its_raw_value(capsys):
    cli_explain._print_coverage(
        {"coverage": 1.0, "random": None, "units": 3, "overshoot": True, "raw": 1.25}
    )
    out = capsys.readouterr().out
    assert "keeping only the 3 explained units" in out and "matched random units" not in out
    assert "Overshoot: the raw coverage is 125.0%." in out


def test_the_record_lists_superpositions_and_circuits_after_the_units():
    from openinterpretability.explain_groups import GroupAttribution

    group = GroupAttribution("circuit", "C4", 3, -2.0, "VERIFIED", "a copy circuit")
    result = SimpleNamespace(total=8.0, poles=[], groups=[group])
    (unit,) = cli_explain._explained_units("toy", result)
    assert (unit.unit, unit.kind, unit.share, unit.attribution) == ("C4", "circuit", 0.25, -2.0)
    assert unit.status == "VERIFIED" and unit.explanation == "a copy circuit"


# The default corpus (#5829): built once into the cache, then reused.
def test_explain_without_corpus_uses_the_default(explain_args, monkeypatch, tmp_path):
    from tests.test_explain_corpus import CORPUS

    hub = []

    def rebuild(source, out):
        hub.append(out)
        out.write_text(CORPUS)
        return source

    monkeypatch.setenv("OPENINTERPRETABILITY_REBUILT_CORPORA", str(tmp_path / "cache"))
    monkeypatch.setattr("openinterpretability.corpus_rebuild.rebuild", rebuild)
    monkeypatch.chdir(tmp_path)  # no corpus.jsonl here
    i = explain_args.index("--corpus")
    args = explain_args[:i] + explain_args[i + 2 :]
    result = CliRunner().invoke(cli, [*args, "--json"])
    assert result.exit_code == 0, result.output
    assert len(hub) == 1
    again = CliRunner().invoke(cli, [*args, "--json"])
    assert again.exit_code == 0 and len(hub) == 1  # the second run reuses the cache


def test_explain_without_a_corpus_or_a_cache_fails_clearly(explain_args, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENINTERPRETABILITY_REBUILT_CORPORA", "off")
    i = explain_args.index("--corpus")
    result = CliRunner().invoke(cli, explain_args[:i] + explain_args[i + 2 :])
    assert result.exit_code == 1
    assert "the corpus cache is off" in result.output
