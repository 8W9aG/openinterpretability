"""``openinterp pause`` and ``openinterp resume``: stop new experiments for a while."""

from __future__ import annotations

import click

from openinterp.cli import cli
from openinterp.core import pause


@cli.command(name="pause")
@click.option("--reason", default="a measurement needs a quiet machine", show_default=True)
@click.option(
    "--for",
    "minutes",
    default=pause.DEFAULT_MINUTES,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Minutes until the pause ends by itself.",
)
@click.option(
    "--status",
    is_flag=True,
    help=f"Only show the pause. Exit {pause.EXIT_PAUSED} if there is one, else 0.",
)
def pause_command(reason, minutes, status):
    """Stop new experiments on this machine; the ones that run go on to the end.

    While the pause lasts, `openinterp next` draws nothing and
    scripts/experiment-loop.sh waits. See docs/experiments.md.
    """
    if status:
        current = pause.read()
        click.echo(current.describe() if current else "New experiments are not paused.")
        raise SystemExit(pause.EXIT_PAUSED if current else 0)
    click.echo(pause.start(reason, minutes).describe())


@cli.command(name="resume")
def resume_command():
    """End a pause that `openinterp pause` started."""
    click.echo("Resumed new experiments." if pause.stop() else "New experiments were not paused.")
