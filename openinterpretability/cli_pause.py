"""Stop new experiments for a while.

The commands are ``open-interpretability pause`` and ``open-interpretability resume``.
"""

from __future__ import annotations

import click

from openinterpretability.cli import cli
from openinterpretability.core import pause


@cli.command(name="pause")
@click.option(
    "--reason",
    default=None,
    help=f"Why new experiments wait. [default: {pause.DEFAULT_REASON}]",
)
@click.option(
    "--for",
    "minutes",
    default=None,
    type=click.FloatRange(min=0, min_open=True),
    help=f"Minutes until the pause ends by itself. [default: {pause.DEFAULT_MINUTES}]",
)
@click.option(
    "--status",
    is_flag=True,
    help=f"Only show the pause. Exit {pause.EXIT_PAUSED} if there is one, else 0.",
)
def pause_command(reason, minutes, status):
    """Stop new experiments on this machine; the ones that run go on to the end.

    While the pause lasts, `open-interpretability next` draws nothing and
    scripts/experiment-loop.sh waits. See docs/experiments.md.

    While a pause stands, a bare `pause` only shows it and exits 75, like
    `--status`, so a check by mistake doesn't extend it (#6839). Give --for or
    --reason to replace a standing pause.
    """
    current = pause.read()
    if status or (current and reason is None and minutes is None):
        click.echo(current.describe() if current else "New experiments are not paused.")
        raise SystemExit(pause.EXIT_PAUSED if current else 0)
    new = pause.start(reason or pause.DEFAULT_REASON, minutes or pause.DEFAULT_MINUTES)
    if current:
        click.echo(f"Replaced the earlier pause (by {current.by}).")
    click.echo(new.describe())


@cli.command(name="resume")
def resume_command():
    """End a pause that `open-interpretability pause` started."""
    click.echo("Resumed new experiments." if pause.stop() else "New experiments were not paused.")
