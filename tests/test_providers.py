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
from openinterpretability.core.legacy_names import keychain_services

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


def _provider(**fields):
    return providers.Provider(name="test", model="m", key_env="TEST_PROVIDER_KEY", **fields)


def test_a_missing_key_without_a_keychain_entry_names_only_the_variable(monkeypatch):
    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    with pytest.raises(ValueError) as caught:
        providers.claude_env(_provider(base_url="http://x"), base={})
    assert str(caught.value) == "no API key for test: set $TEST_PROVIDER_KEY"


def test_an_anthropic_provider_with_a_key_passes_it_as_the_api_key(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", SECRET)
    env = providers.claude_env(_provider(), base={})
    assert env["ANTHROPIC_API_KEY"] == SECRET and "ANTHROPIC_BASE_URL" not in env


def test_litellm_env_sets_the_key_only_when_there_is_one(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", SECRET)
    assert providers.litellm_env(_provider(), base={"A": "1"}) == {
        "A": "1", "TEST_PROVIDER_KEY": SECRET,
    }  # fmt: skip
    monkeypatch.delenv("TEST_PROVIDER_KEY")
    assert providers.litellm_env(_provider(), base={"A": "1"}) == {"A": "1"}


def test_listening_tells_an_open_port_from_a_closed_one():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert providers._listening(port)
    assert not providers._listening(port)


def test_start_proxy_does_nothing_when_the_proxy_already_listens(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "_listening", lambda port: True)
    monkeypatch.setattr(providers.subprocess, "Popen", lambda *a, **k: pytest.fail("started"))
    provider = _provider(proxy_config=str(tmp_path / "none.yaml"), proxy_port=4000)
    providers.start_proxy(provider, {}, tmp_path / "proxy.log")
    assert not (tmp_path / "proxy.log").exists()


def test_start_proxy_needs_its_config_and_the_litellm_command(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "_listening", lambda port: False)
    config = tmp_path / "proxy.yaml"
    provider = _provider(proxy_config=str(config), proxy_port=4000)
    with pytest.raises(ValueError, match=f"test needs a LiteLLM proxy config at {config}"):
        providers.start_proxy(provider, {}, tmp_path / "proxy.log")
    config.write_text("model_list: []\n")
    monkeypatch.setattr(providers.shutil, "which", lambda _: None)
    needs = r"needs the LiteLLM proxy: pip install 'litellm\[proxy\]'"
    with pytest.raises(ValueError, match=needs):
        providers.start_proxy(provider, {}, tmp_path / "proxy.log")


def test_start_proxy_starts_litellm_and_waits_until_it_listens(monkeypatch, tmp_path):
    answers = iter([False, False, True])  # down, still starting, then up
    monkeypatch.setattr(providers, "_listening", lambda port: next(answers))
    monkeypatch.setattr(providers.shutil, "which", lambda _: "/usr/bin/litellm")
    sleeps, started = [], []
    monkeypatch.setattr(providers.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        providers.subprocess, "Popen", lambda args, **kw: started.append((args, kw["env"]))
    )
    config = tmp_path / "proxy.yaml"
    config.write_text("model_list: []\n")
    log = tmp_path / "proxy.log"
    providers.start_proxy(_provider(proxy_config=str(config), proxy_port=4001), {"K": "v"}, log)
    assert started == [
        (
            ["litellm", "--config", str(config), "--host", "127.0.0.1", "--port", "4001"],
            {"K": "v"},
        )
    ]
    assert sleeps == [0.5] and log.exists()


def test_start_proxy_raises_when_the_proxy_never_listens(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "_listening", lambda port: False)
    monkeypatch.setattr(providers.shutil, "which", lambda _: "/usr/bin/litellm")
    monkeypatch.setattr(providers.subprocess, "Popen", lambda *a, **k: None)
    config = tmp_path / "proxy.yaml"
    config.write_text("model_list: []\n")
    log = tmp_path / "proxy.log"
    provider = _provider(proxy_config=str(config), proxy_port=4002)
    with pytest.raises(ValueError, match=f"the test proxy did not start; see {log}"):
        providers.start_proxy(provider, {}, log, wait_s=0.0)


def test_annotate_leaves_unreadable_and_non_object_sessions_alone(tmp_path):
    provider = providers.load("deepseek")
    providers.annotate(tmp_path / "missing.json", provider)
    assert not (tmp_path / "missing.json").exists()
    for text in ("not json", "[1, 2]"):
        session = tmp_path / "s.json"
        session.write_text(text)
        providers.annotate(session, provider)
        assert session.read_text() == text


def test_the_old_keychain_name_is_tried_and_none_when_neither_has_the_key(monkeypatch):
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    monkeypatch.setattr(providers.shutil, "which", lambda _: "/usr/bin/security")
    services = []
    new, old = keychain_services("openinterpretability-muse")

    def fake_run(args, **_):
        services.append(args[3])
        if args[3] == old:  # found under the old name, but empty
            return subprocess.CompletedProcess(args, 0, stdout="  \n", stderr="")
        return subprocess.CompletedProcess(args, 44, stdout="", stderr="not found")

    monkeypatch.setattr(providers.subprocess, "run", fake_run)
    assert providers.key(providers.load("muse")) is None
    assert services == [new, old]
