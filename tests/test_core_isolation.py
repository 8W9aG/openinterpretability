"""The reviewer isolation hook, run in this process: it denies blocked calls and fails closed."""

import io
import json
import runpy
import sys
import warnings

import pytest

from openinterpretability.core import isolation


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "submissions").mkdir()
    (tmp_path / "exchange/k").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    return tmp_path


def _run(monkeypatch, capsys, stdin: str):
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    assert isolation.main() == 0
    out = capsys.readouterr().out
    return json.loads(out)["hookSpecificOutput"] if out else None


def test_the_hook_denies_a_read_of_a_submission(repo, monkeypatch, capsys):
    event = {"tool_name": "Read", "tool_input": {"file_path": "submissions/s"}, "cwd": str(repo)}
    decision = _run(monkeypatch, capsys, json.dumps(event))
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    reason = decision["permissionDecisionReason"]
    assert reason.startswith("Reviewer isolation (docs/PROTOCOL.md §2): submissions/s is blocked.")


def test_the_hook_allows_a_read_of_the_prompt_and_prints_nothing(repo, monkeypatch, capsys):
    event = {"tool_name": "Read", "tool_input": {"file_path": "exchange/k/review.prompt.md"}}
    monkeypatch.chdir(repo)  # an event with no cwd uses the current directory
    assert _run(monkeypatch, capsys, json.dumps(event)) is None


def test_the_hook_denies_when_the_event_is_not_json(repo, monkeypatch, capsys):
    decision = _run(monkeypatch, capsys, "not json")
    assert decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"].startswith(
        "The reviewer isolation check failed (JSONDecodeError("
    )


def test_running_the_module_exits_with_the_hooks_status(repo, monkeypatch, capsys):
    event = {"tool_name": "Grep", "tool_input": {"path": "submissions"}, "cwd": str(repo)}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the module is already imported
        with pytest.raises(SystemExit) as exit_:
            runpy.run_module("openinterpretability.core.isolation", run_name="__main__")
    assert exit_.value.code == 0
    decision = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert f"{repo / 'submissions'} is blocked" in decision["permissionDecisionReason"]
