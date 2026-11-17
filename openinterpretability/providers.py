"""Model providers that can drive an experiment loop (issue #3245, docs/providers.md).

A provider profile, in ``openinterpretability/data/providers.toml``, says how to point Claude Code
(and LiteLLM) at a model provider. It gives the base URL, the model names, extra
environment, and where the API key lives. It never holds the key. :func:`key` reads it
from an environment variable or the macOS keychain. :func:`claude_env` puts it only in
the environment of the command that needs it.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class Provider:
    """One profile from ``providers.toml``."""

    name: str
    model: str
    key_env: str
    base_url: str = ""
    small_model: str = ""
    key_keychain: str = ""
    key_required: bool = True
    litellm_model: str = ""
    proxy_config: str = ""
    proxy_port: int = 0
    reported_model: str = ""
    env: dict[str, str] = field(default_factory=dict)

    @property
    def driver(self) -> str:
        """The model to record as the one that drove a session."""
        return self.reported_model or self.model

    @property
    def anthropic(self) -> bool:
        """True when Claude Code talks to Anthropic itself, so its own prices hold."""
        return not self.base_url


def profiles() -> dict[str, Provider]:
    """Every profile, by name."""
    text = files("openinterpretability.data").joinpath("providers.toml").read_text(encoding="utf-8")
    return {name: Provider(name=name, **fields) for name, fields in tomllib.loads(text).items()}


def load(name: str) -> Provider:
    """The profile called ``name``; a ValueError names the known ones."""
    known = profiles()
    if name not in known:
        raise ValueError(f"unknown provider {name!r}; known: {', '.join(sorted(known))}")
    return known[name]


def key(provider: Provider) -> str | None:
    """The provider's API key, from ``key_env`` or else the keychain; None when neither has it."""
    if value := os.environ.get(provider.key_env, "").strip():
        return value
    from openinterpretability.core.legacy_names import keychain_services

    if not (provider.key_keychain and shutil.which("security")):
        return None
    for service in keychain_services(provider.key_keychain):  # the old name too (#4089)
        done = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout.strip()
    return None


def claude_env(provider: Provider, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for Claude Code to drive with ``provider``.

    A ValueError says where to put the key when a provider needs one and has none.
    """
    env = dict(os.environ if base is None else base)
    secret = key(provider)
    if secret is None and provider.key_required:
        where = f"${provider.key_env}"
        if provider.key_keychain:
            where += f" or the keychain entry {provider.key_keychain!r}"
        raise ValueError(f"no API key for {provider.name}: set {where}")
    small = provider.small_model or provider.model
    if provider.base_url:
        env["ANTHROPIC_BASE_URL"] = provider.base_url
        env["ANTHROPIC_MODEL"] = provider.model
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = provider.model
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = provider.model
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = small
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = small
        # A LiteLLM proxy with no master key accepts any token. The proxy itself reads
        # the provider's key from key_env.
        env["ANTHROPIC_AUTH_TOKEN"] = "sk-local-proxy" if provider.proxy_config else secret
        if provider.proxy_config and secret:
            env[provider.key_env] = secret
    elif secret:
        env["ANTHROPIC_API_KEY"] = secret
    env.update(provider.env)
    env["OPENINTERPRETABILITY_DRIVER_PROVIDER"] = provider.name
    env["OPENINTERPRETABILITY_DRIVER_MODEL"] = provider.driver
    return env


def litellm_env(provider: Provider, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for LiteLLM calls to ``provider``: its key under ``key_env``."""
    env = dict(os.environ if base is None else base)
    if (secret := key(provider)) is not None:
        env[provider.key_env] = secret
    return env


def _listening(port: int) -> bool:
    """True when something accepts connections on 127.0.0.1:``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_proxy(provider: Provider, env: dict[str, str], log: Path, wait_s: float = 8.0) -> None:
    """Start the provider's LiteLLM proxy unless it already listens. Raise if it doesn't start."""
    if not provider.proxy_config or _listening(provider.proxy_port):
        return
    config = Path(provider.proxy_config).expanduser()
    if not config.is_file():
        raise ValueError(f"{provider.name} needs a LiteLLM proxy config at {config}")
    if not shutil.which("litellm"):
        raise ValueError(f"{provider.name} needs the LiteLLM proxy: pip install 'litellm[proxy]'")
    with log.open("ab") as out:
        subprocess.Popen(
            ["litellm", "--config", str(config), "--host", "127.0.0.1",
             "--port", str(provider.proxy_port)],
            env=env, stdout=out, stderr=out, start_new_session=True,
        )  # fmt: skip
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _listening(provider.proxy_port):
            return
        time.sleep(0.5)
    raise ValueError(f"the {provider.name} proxy did not start; see {log}")


def annotate(session: Path, provider: Provider) -> None:
    """Record the real driver in a ``claude --output-format json`` session file.

    Claude Code prices every model as an Anthropic one, so for any other provider the
    session's dollar cost is wrong. It is kept as ``claude_code_cost_usd`` and the cost
    becomes unpriced (None), which ``open-interpretability costs`` reports as such.
    """
    try:
        data = json.loads(session.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    data["driver"] = {"provider": provider.name, "model": provider.driver}
    if not provider.anthropic and data.get("total_cost_usd") is not None:
        data["claude_code_cost_usd"] = data["total_cost_usd"]
        data["total_cost_usd"] = None
    session.write_text(json.dumps(data), encoding="utf-8")
