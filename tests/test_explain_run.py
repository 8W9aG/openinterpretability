"""``explain --run``: the run's target and the agents it hands to the loop."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterpretability import explain as ex
from openinterpretability import providers
from openinterpretability.explain_run import _reachable
from tests.test_explain import PROMPT, TEXTS


def test_an_unreachable_target_stops_at_the_reach_and_names_waiting_circuits(capsys):
    reach = {"now": 0.1, "reach": 0.4, "target": 0.8, "experiments": None}
    planned = SimpleNamespace(blocked=0.05)
    assert _reachable(reach, planned, ["C1 (a task to write)"]) == 0.4
    out = capsys.readouterr().out
    assert "Not run: circuit C1 (a task to write)" in out
    assert "The target 80% is out of reach, so the run stops at 40.0%" in out


def test_a_run_with_only_a_provider_takes_the_providers_model(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    seen = {}

    def run_loop(proposer, reviewer, plan_next, out=None, local=False, sessions=None):
        seen.update(proposer=proposer, reviewer=reviewer)
        return 0

    monkeypatch.setattr("openinterpretability.cli_loop.run_loop", run_loop)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(tmp_path / "wiki"), "--edges", "0", "--run", "1", "--yes",
            "--provider", "anthropic"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    model = providers.load("anthropic").litellm_model
    assert model and seen == {"proposer": model, "reviewer": model}


# #5831: with no LiteLLM model, `explain --run` picks Claude Code sessions, or says why not.


def _run_args(toy_runner, tmp_path, monkeypatch, *extra):
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)  # no API key at all
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    return [
        "explain",
        PROMPT,
        "--model",
        "toy",
        "--corpus",
        str(corpus),
        "--mean-tokens",
        "200",
        "--wiki-dir",
        str(tmp_path / "wiki"),
        "--edges",
        "0",
        "--run",
        "2",
        "--yes",
        *extra,
    ]


def _claude_on_path(tmp_path, monkeypatch):
    path = tmp_path / "claude"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    monkeypatch.setenv("CLAUDE_BIN", str(path))


def test_a_run_with_no_model_and_claude_on_path_runs_sessions(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli
    from openinterpretability.loop_sessions import SUBAGENTS

    _claude_on_path(tmp_path, monkeypatch)
    monkeypatch.setattr("openinterpretability.cli_loop.loop_worktree",
                        lambda *a: (tmp_path / "repo", tmp_path / "wt"))  # fmt: skip
    seen = {}

    def run_loop(proposer, reviewer, plan_next, out=None, local=False, sessions=None):
        seen.update(agents=(proposer, reviewer), sessions=sessions)
        return 0

    monkeypatch.setattr("openinterpretability.cli_loop.run_loop", run_loop)
    out = CliRunner().invoke(cli, _run_args(toy_runner, tmp_path, monkeypatch))
    assert out.exit_code == 0, out.output
    assert seen["agents"] == (SUBAGENTS, SUBAGENTS)
    assert seen["sessions"].log_dir == tmp_path / "repo" / "logs" / "explain"
    assert "headless Claude Code session" in out.output


@pytest.mark.parametrize(
    ("claude_code", "says"),
    [(False, "or install Claude Code"), (True, "run the explain skill")],
)
def test_a_run_with_no_model_and_no_claude_says_what_to_do(toy_runner, tmp_path, monkeypatch,
                                                            claude_code, says):  # fmt: skip
    from openinterpretability.cli import cli

    args = _run_args(toy_runner, tmp_path, monkeypatch)
    monkeypatch.setenv("CLAUDE_BIN", str(tmp_path / "no-such-claude"))
    if claude_code:
        monkeypatch.setenv("CLAUDECODE", "1")
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 2 and says in out.output


def test_local_with_sessions_is_refused(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli

    _claude_on_path(tmp_path, monkeypatch)
    out = CliRunner().invoke(cli, _run_args(toy_runner, tmp_path, monkeypatch, "--local"))
    assert out.exit_code == 2 and "--local needs a LiteLLM model" in out.output
