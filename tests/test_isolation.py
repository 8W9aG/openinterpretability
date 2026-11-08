"""Reviewer isolation: the hook's decisions, the transcript check and `review --probes`."""

import json
import subprocess
import sys

import click
import pytest
from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.cli_review import refuse_unisolated
from openinterp.core import costs
from openinterp.core.isolation import forbidden
from openinterp.core.transcripts import forbidden_reads


@pytest.fixture
def repo(tmp_path):
    for d in ("wiki/experiments/m/L1/N2-positive", "exchange/k", "submissions", ".claude/skills"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "exchange/k/review.prompt.md").write_text("prompt")
    return tmp_path


@pytest.mark.parametrize(
    ("tool", "tool_input", "denied"),
    [
        ("Read", {"file_path": "wiki/experiments/m/L1/N2-positive/r.json"}, True),
        ("Read", {"file_path": "submissions/s.json"}, True),
        ("Read", {"file_path": "exchange/k/review.log"}, True),
        ("Read", {"file_path": "exchange/k/ev.json"}, True),
        ("Read", {"file_path": "logs/loop/x.log"}, True),
        ("Read", {"file_path": "dist/index.html"}, True),
        ("Read", {"file_path": "wiki/experiments/../experiments/m/r.json"}, True),
        ("Read", {"file_path": "exchange/k/review.prompt.md"}, False),
        ("Read", {"file_path": "exchange/k/review.check.prompt.md"}, False),
        ("Write", {"file_path": "exchange/k/review.probes.json"}, False),
        ("Write", {"file_path": "exchange/k/check.py"}, True),
        ("Read", {"file_path": ".claude/skills/x/SKILL.md"}, False),
        ("Read", {"file_path": "/elsewhere/review.prompt.md"}, False),
        ("Grep", {"pattern": "x"}, True),  # no path: the whole project
        ("Grep", {"pattern": "x", "path": "wiki"}, True),
        ("Grep", {"pattern": "x", "path": "exchange/k"}, True),
        ("Grep", {"pattern": "x", "path": "exchange/k/review.probes.json"}, False),
        ("Grep", {"pattern": "x", "path": ".claude/skills"}, False),
        ("Glob", {"pattern": "**/*"}, False),  # names only
    ],
)
def test_forbidden(repo, tool, tool_input, denied):
    assert (forbidden(tool, tool_input, repo, repo) is not None) is denied


def test_symlink_into_a_blocked_dir_is_denied(repo):
    (repo / ".claude/skills/link").symlink_to(repo / "wiki/experiments")
    assert forbidden("Read", {"file_path": ".claude/skills/link/r.json"}, repo, repo)


def _hook(event: dict, root) -> dict | None:
    out = subprocess.run(
        [sys.executable, "-m", "openinterp.core.isolation"],
        input=json.dumps(event) if isinstance(event, dict) else event,
        capture_output=True,
        text=True,
        env={"CLAUDE_PROJECT_DIR": str(root), "PATH": ""},
        check=True,
    ).stdout
    return json.loads(out)["hookSpecificOutput"] if out else None


def test_hook_denies_blocked_calls_and_fails_closed(repo):
    cwd = str(repo)
    denied = _hook(
        {"tool_name": "Read", "tool_input": {"file_path": "submissions/s"}, "cwd": cwd}, repo
    )
    assert (
        denied["permissionDecision"] == "deny"
        and "submissions/s" in denied["permissionDecisionReason"]
    )
    ok = {
        "tool_name": "Read",
        "tool_input": {"file_path": "exchange/k/review.prompt.md"},
        "cwd": cwd,
    }
    assert _hook(ok, repo) is None
    assert _hook("not json", repo)["permissionDecision"] == "deny"


def _session(path, calls):
    """A transcript in which each (tool, input, is_error) call has its result."""
    lines = []
    for i, (tool, tool_input, error) in enumerate(calls):
        use = {"type": "tool_use", "id": f"t{i}", "name": tool, "input": tool_input}
        result = {"type": "tool_result", "tool_use_id": f"t{i}", "is_error": error}
        lines.append({"type": "assistant", "cwd": str(path.parent), "message": {"content": [use]}})
        lines.append({"type": "user", "cwd": str(path.parent), "message": {"content": [result]}})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n{cut off\n")
    return path


def test_forbidden_reads_splits_completed_from_denied(repo):
    transcript = _session(
        repo / "agent-a.jsonl",
        [
            ("Read", {"file_path": "exchange/k/review.prompt.md"}, False),
            ("Read", {"file_path": "submissions/s.json"}, True),  # denied by the hook
            ("Grep", {"pattern": "x"}, False),
            ("Grep", {"pattern": "y"}, False),
        ],
    )
    assert forbidden_reads(transcript) == ([str(repo)], ["submissions/s.json"])


def test_usage_records_forbidden_reads_and_review_refuses(repo, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(repo / "claude"))
    sessions = repo / "claude/projects/p/s/subagents"
    sessions.mkdir(parents=True)
    _session(sessions / "agent-bad.jsonl", [("Read", {"file_path": "wiki/experiments/m/r"}, False)])
    _session(
        sessions / "agent-good.jsonl",
        [("Read", {"file_path": "exchange/k/review.prompt.md"}, False)],
    )
    probes = repo / "exchange/k/review.probes.json"
    probes.write_text("{}")

    result = CliRunner().invoke(cli, ["usage", "good", "--answer", str(probes)])
    assert result.exit_code == 0, result.output
    refuse_unisolated(probes)

    result = CliRunner().invoke(cli, ["usage", "bad", "--answer", str(probes)])
    assert result.exit_code == 0 and "will refuse" in result.output
    usage = costs.ExternalUsage.model_validate_json(costs.usage_path(probes).read_text())
    assert usage.forbidden_reads == ["wiki/experiments/m/r"]
    with pytest.raises(click.ClickException, match="read blocked paths"):
        refuse_unisolated(probes)
