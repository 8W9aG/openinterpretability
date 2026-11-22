"""``open-interpretability claude-session``: one headless Claude Code session (#5831)."""

from __future__ import annotations

from pathlib import Path

import click

from openinterpretability.claude_sessions import DRIVER, MAX_TURNS, Session
from openinterpretability.cli import cli


@cli.command(name="claude-session")
@click.option(
    "--driver", default=DRIVER, show_default=True, help="The session's model; '' for claude's own."
)
@click.option("--max-turns", default=MAX_TURNS, show_default=True, type=int)
@click.option("--provider", default="", help="A provider profile whose environment it runs in.")
@click.option("--result", type=click.Path(path_type=Path), required=True, help="The JSON result.")
@click.option("--log", type=click.Path(path_type=Path), required=True, help="Its stderr.")
@click.option("--ledger", type=click.Path(path_type=Path), required=True, help="Its costs ledger.")
def _claude_session(driver, max_turns, provider, result, log, ledger):
    """Run one headless Claude Code session on the prompt read from stdin, in this folder.

    It exits with claude's exit code. scripts/experiment-loop.sh and `explain --run` start
    their sessions this way (openinterpretability/claude_sessions.py).

    \f
    Args:
        driver: The session's model.
        max_turns: Its turns at most.
        provider: A provider profile, or "".
        result: Where the JSON result goes.
        log: Where stderr goes.
        ledger: The costs ledger.
    """
    prompt = click.get_text_stream("stdin").read()
    session = Session(driver, max_turns, provider)
    raise SystemExit(session.run(prompt, Path.cwd(), result, log, ledger))
