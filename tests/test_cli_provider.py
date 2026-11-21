"""``open-interpretability provider``: model, exec and annotate, run in this process (#3245)."""

import json

from click.testing import CliRunner

from openinterpretability import cli_provider, providers
from openinterpretability.cli import cli

SECRET = "sk-test-not-a-real-key"


def test_model_prints_the_profiles_driving_model():
    result = CliRunner().invoke(cli, ["provider", "model", "deepseek"])
    assert result.exit_code == 0, result.output
    assert result.output == "deepseek-flash[1m]\n"


def test_an_unknown_provider_is_a_usage_error_that_names_the_known_ones():
    result = CliRunner().invoke(cli, ["provider", "model", "nobody"])
    assert result.exit_code == 2
    assert "unknown provider 'nobody'; known: " in result.output
    assert "deepseek" in result.output


def test_exec_exits_with_the_commands_return_code(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    calls = []

    class Done:
        returncode = 3

    def run(command, env, check):
        calls.append((command, env))
        return Done()

    monkeypatch.setattr(cli_provider.subprocess, "run", run)
    args = ["provider", "exec", "deepseek", "--", "claude", "-p", "hi"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 3
    ((command, env),) = calls
    assert command == ["claude", "-p", "hi"]
    assert env["ANTHROPIC_AUTH_TOKEN"] == SECRET
    assert SECRET not in result.output


def test_exec_with_only_a_separator_asks_for_a_command(monkeypatch):
    monkeypatch.setattr(cli_provider.subprocess, "run", lambda *a, **k: 1 / 0)
    # Click passes the "--" on as the whole command, and exec drops it.
    result = CliRunner().invoke(cli, ["provider", "exec", "deepseek", "--"])
    assert result.exit_code == 2
    assert "give a command to run after NAME" in result.output


def test_exec_without_the_providers_key_fails_and_runs_nothing(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(providers.shutil, "which", lambda _: None)
    monkeypatch.setattr(cli_provider.subprocess, "run", lambda *a, **k: 1 / 0)
    result = CliRunner().invoke(cli, ["provider", "exec", "zai", "--", "claude"])
    assert result.exit_code == 1
    assert "$ZAI_API_KEY" in result.output


def test_annotate_records_the_driver_in_the_session_file(tmp_path):
    session = tmp_path / "s.json"
    session.write_text(json.dumps({"total_cost_usd": 1.25}))
    result = CliRunner().invoke(cli, ["provider", "annotate", "deepseek", str(session)])
    assert result.exit_code == 0, result.output
    data = json.loads(session.read_text())
    assert data["driver"] == {"provider": "deepseek", "model": "deepseek-flash[1m]"}
    assert data["total_cost_usd"] is None
