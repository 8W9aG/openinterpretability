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
    assert as_json["target"] == 0.5 and as_json["reaches"] >= 0.5 and "order" not in as_json
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

    def run_loop(proposer, reviewer, plan_next):
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
    assert len(draws.targets) == 2 and "Running 2 unit experiments" in out.output
    assert all("--selection attribution" in t.audit for t in draws.targets)
    external = CliRunner().invoke(
        cli, [*args[:-4], "--proposer", "external/x", "--reviewer", "anthropic/r"]
    )
    assert external.exit_code != 0  # nobody would answer an external model's prompt files


def test_cli_records_the_explanation(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli
    from openinterpretability.core.explanation_records import load_latest

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    wiki = tmp_path / "wiki"
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(wiki), "--edges", "0", "--record"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    ((path, record),) = load_latest(wiki)
    assert f"Explanation → {path}" in out.output
    assert record.model_name == "toy" and record.prompt == PROMPT and record.narration is None
    assert record.units and record.planned_experiments > 0
