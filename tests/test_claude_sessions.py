"""Headless Claude Code sessions and the Claude Code defaults of `explain` (#5831)."""

import json
import sys
from pathlib import Path

from click.testing import CliRunner

from openinterpretability.claude_sessions import ALLOWED_TOOLS, Session
from openinterpretability.cli import cli
from openinterpretability.explain_backend import narrator

ROOT = Path(__file__).resolve().parents[1]
FAKE = """#!{python}
import json, os, sys
open(os.environ["CLAUDE_LOG"], "w").write(json.dumps({{
    "args": sys.argv[1:], "prompt": sys.stdin.read(),
    "ledger": os.environ.get("OPENINTERPRETABILITY_COSTS_LOG")}}))
print('{{"result": "RESULT: none tested"}}')
print("to stderr", file=sys.stderr)
sys.exit(4)
"""


def test_the_command_runs_one_session_on_stdin_and_exits_with_claudes_code(tmp_path,
                                                                          monkeypatch):  # fmt: skip
    claude = tmp_path / "claude"
    claude.write_text(FAKE.format(python=sys.executable))
    claude.chmod(0o755)
    monkeypatch.setenv("CLAUDE_BIN", str(claude))
    monkeypatch.setenv("CLAUDE_LOG", str(tmp_path / "seen.json"))
    monkeypatch.chdir(tmp_path)
    args = ["claude-session", "--driver", "", "--max-turns", "9", "--result", "out.json",
            "--log", "err.log", "--ledger", "costs.jsonl"]  # fmt: skip
    result = CliRunner().invoke(cli, args, input="Run one experiment.")
    assert result.exit_code == 4
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["prompt"] == "Run one experiment." and seen["ledger"] == "costs.jsonl"
    assert seen["args"][:6] == ["-p", "--output-format", "json", "--permission-mode",
                                "acceptEdits", "--max-turns"]  # fmt: skip
    assert "--model" not in seen["args"]  # '' leaves claude's own model
    assert seen["args"][-len(ALLOWED_TOOLS) :] == list(ALLOWED_TOOLS)
    assert json.loads((tmp_path / "out.json").read_text())["result"] == "RESULT: none tested"
    assert (tmp_path / "err.log").read_text() == "to stderr\n"


def test_a_provider_wraps_claude_in_its_environment():
    command = Session("m", 3, "deepseek").command({"CLAUDE_BIN": "/c"})
    assert command[:9] == ["uv", "run", "--quiet", "open-interpretability", "provider", "exec",
                           "deepseek", "--", "/c"]  # fmt: skip


def test_the_loop_script_launches_its_sessions_through_the_command():
    script = (ROOT / "scripts" / "experiment-loop.sh").read_text()
    assert "open-interpretability claude-session" in script
    assert "ALLOWED_TOOLS" not in script and "claude_args" not in script  # one launcher


def test_inside_claude_code_the_narrator_is_a_subagent(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("OPENINTERPRETABILITY_EXCHANGE_DIR", str(tmp_path / "ex"))
    said = []
    assert narrator(None, said.append) == "external/claude-code"
    assert said == [f"The narrator is external/claude-code: answer {tmp_path / 'ex'}"
                    "/Narrator.prompt.md with the explanation-narrator agent."]  # fmt: skip
    monkeypatch.delenv("CLAUDECODE")
    assert narrator(None, said.append) is None and len(said) == 1
