"""``open-interpretability provider``: drive Claude Code with another model provider (#3245).

``scripts/experiment-loop.sh --provider NAME`` runs each session through ``provider exec``.
See docs/providers.md.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import click

from openinterpretability import providers
from openinterpretability.cli import cli


def _load(name: str) -> providers.Provider:
    try:
        return providers.load(name)
    except ValueError as error:
        raise click.UsageError(str(error)) from None


@cli.group(name="provider")
def provider_group() -> None:
    """Model providers that can drive a session: profiles in data/providers.toml."""


@provider_group.command(name="list")
def list_command() -> None:
    """Each profile, its driving model, and whether its key is set (never the key)."""
    for name, provider in sorted(providers.profiles().items()):
        has_key = "key set" if providers.key(provider) else f"no key (${provider.key_env})"
        via = f", proxy :{provider.proxy_port}" if provider.proxy_config else ""
        click.echo(f"{name}: {provider.driver}{via}; {has_key}")


@provider_group.command(name="model")
@click.argument("name")
def model_command(name: str) -> None:
    """Print the model that Claude Code drives with for NAME (its `--model`)."""
    click.echo(_load(name).model)


@provider_group.command(
    name="exec", context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False}
)
@click.argument("name")
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
def exec_command(name: str, command: tuple[str, ...]) -> None:
    """Run COMMAND (e.g., `claude -p …`) with NAME's environment, starting its proxy if needed.

    The key goes into COMMAND's environment only. Nothing prints it.
    """
    provider = _load(name)
    command = command[1:] if command[:1] == ("--",) else command
    if not command:
        raise click.UsageError("give a command to run after NAME")
    try:
        env = providers.claude_env(provider)
        log = Path(tempfile.gettempdir()) / f"openinterpretability-litellm-{name}.log"
        providers.start_proxy(provider, env, log)
    except ValueError as error:
        raise click.ClickException(str(error)) from None
    # The child inherits stdin and stdout, so a prompt piped in reaches it.
    raise SystemExit(subprocess.run(list(command), env=env, check=False).returncode)


@provider_group.command(name="annotate")
@click.argument("name")
@click.argument("session", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def annotate_command(name: str, session: Path) -> None:
    """Record NAME as the driver in SESSION, a `claude --output-format json` file."""
    providers.annotate(session, _load(name))
