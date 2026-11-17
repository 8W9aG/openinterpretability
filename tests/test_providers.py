"""Provider profiles that drive an experiment loop (openinterpretability/providers.py, #3245)."""

import json
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from openinterpretability import cli_loop, providers
from openinterpretability.cli import cli
from openinterpretability.cli_costs import _session

SECRET = "sk-test-not-a-real-key"


def test_every_profile_loads_and_needs_no_secret_in_the_repo():
    known = providers.profiles()
    assert {"anthropic", "deepseek", "zai", "muse", "gemini", "grok"} <= set(known)
    for provider in known.values():
        assert provider.key_env and provider.model
        assert "sk-" not in json.dumps(provider.__dict__)


def test_deepseek_gets_its_url_model_fixes_and_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    env = providers.claude_env(providers.load("deepseek"), base={})
    assert env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert env["ANTHROPIC_MODEL"] == "deepseek-flash[1m]"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "deepseek-flash"
    assert env["ANTHROPIC_AUTH_TOKEN"] == SECRET
    assert env["ENABLE_TOOL_SEARCH"] == "false"
    assert int(env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]) < 1_000_000
    assert env["OPENINTERPRETABILITY_DRIVER_MODEL"] == "deepseek-flash[1m]"


def test_a_missing_key_names_where_to_put_it_and_never_a_value(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(providers.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match=r"\$ZAI_API_KEY"):
        providers.claude_env(providers.load("zai"), base={})


def test_anthropic_needs_no_key_and_sets_no_base_url(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(providers.shutil, "which", lambda _: None)
    env = providers.claude_env(providers.load("anthropic"), base={})
    assert "ANTHROPIC_BASE_URL" not in env and "ANTHROPIC_AUTH_TOKEN" not in env


def test_a_proxy_provider_gives_claude_a_local_token_and_the_proxy_the_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", SECRET)
    provider = providers.load("grok")
    env = providers.claude_env(provider, base={})
    assert env["ANTHROPIC_AUTH_TOKEN"] != SECRET and env["XAI_API_KEY"] == SECRET
    assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{provider.proxy_port}"
    assert env["OPENINTERPRETABILITY_DRIVER_MODEL"].startswith("xai/")


def test_the_key_comes_from_the_keychain_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    monkeypatch.setattr(providers.shutil, "which", lambda _: "/usr/bin/security")
    calls = []

    def fake_run(args, **_):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=SECRET + "\n", stderr="")

    monkeypatch.setattr(providers.subprocess, "run", fake_run)
    assert providers.key(providers.load("muse")) == SECRET
    assert calls[0][:3] == ["security", "find-generic-password", "-s"]


def test_exec_runs_the_command_with_the_environment_and_prints_no_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    script = (
        "import os; env = os.environ; "
        "print(env['ANTHROPIC_MODEL'], len(env['ANTHROPIC_AUTH_TOKEN']))"
    )
    command = str(Path(sys.executable).parent / "open-interpretability")
    done = subprocess.run(
        [command, "provider", "exec", "deepseek", "--",
         sys.executable, "-c", script],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["deepseek-flash[1m]", str(len(SECRET))]
    assert SECRET not in done.stdout + done.stderr


def test_annotate_records_the_driver_and_unprices_a_foreign_session(tmp_path):
    session = tmp_path / "s.json"
    session.write_text(json.dumps({"total_cost_usd": 1.25, "num_turns": 3}))
    providers.annotate(session, providers.load("deepseek"))
    data = json.loads(session.read_text())
    assert data["driver"] == {"provider": "deepseek", "model": "deepseek-flash[1m]"}
    assert data["total_cost_usd"] is None and data["claude_code_cost_usd"] == 1.25
    assert "driven by deepseek/deepseek-flash[1m]" in _session([data])


def test_annotate_keeps_an_anthropic_sessions_cost(tmp_path):
    session = tmp_path / "s.json"
    session.write_text(json.dumps({"total_cost_usd": 0.5}))
    providers.annotate(session, providers.load("anthropic"))
    assert json.loads(session.read_text())["total_cost_usd"] == 0.5


def test_the_loop_takes_its_models_from_the_profile(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    assert cli_loop._models("", "", "deepseek") == ("deepseek/deepseek-flash",) * 2
    models = cli_loop._models("", "x/reviewer", "deepseek")
    assert models == ("deepseek/deepseek-flash", "x/reviewer")
    with pytest.raises(click.UsageError, match="no LiteLLM model"):
        cli_loop._models("", "", "zai")
    with pytest.raises(click.UsageError):
        cli_loop._models("", "", "")


def test_provider_list_says_whether_a_key_is_set_but_never_shows_it(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    result = CliRunner().invoke(cli, ["provider", "list"])
    assert result.exit_code == 0 and "deepseek: deepseek-flash[1m]; key set" in result.output
    assert SECRET not in result.output
