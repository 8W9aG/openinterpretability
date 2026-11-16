"""Run experiments back to back from the CLI, with no Claude session.

The command is ``open-interpretability loop``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click

from openinterp.cli import cli
from openinterp.core.agent_names import EXTERNAL_PREFIX


def _api_model(_ctx, param, value: str) -> str:
    """Refuse ``external/<name>``: nobody is there to answer its prompt files."""
    if value.startswith(EXTERNAL_PREFIX):
        raise click.BadParameter(
            f"{value} needs someone to answer prompt files; use scripts/experiment-loop.sh, "
            "which drives Claude Code subagents, or give a LiteLLM model.",
            param=param,
        )
    return value


@cli.command(name="loop")
@click.option(
    "--proposer", default="", callback=_api_model,
    help="LiteLLM proposer model. Default: the --provider profile's model.",
)  # fmt: skip
@click.option(
    "--reviewer",
    default="",
    callback=_api_model,
    help="LiteLLM reviewer model. A review draw of this model's own record is drawn again. "
    "Default: the --provider profile's model.",
)
@click.option(
    "--provider",
    default="",
    help="A profile in openinterp/data/providers.toml (docs/providers.md): its LiteLLM model "
    "and its API key, read from the environment or the keychain.",
)
@click.option(
    "--max",
    "max_experiments",
    default=0,
    show_default=True,
    help="Stop after N experiments; 0 runs until stopped.",
)
@click.option(
    "--epsilon",
    default=0.7,
    show_default=True,
    type=click.FloatRange(0, 1),
    help="Exploration probability, as in `open-interpretability next`.",
)
@click.option(
    "--corpus",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Scan corpus. Default: each new scan builds a fresh `mixed` corpus.",
)
@click.option(
    "--worktree",
    type=click.Path(file_okay=False, path_type=Path),
    help="Worktree to run in. Default: ../<repo>-loop, created if missing.",
)
@click.option(
    "--one-heavy-job",
    is_flag=True,
    help="Queue behind a machine-wide lock before loading the model (openinterp/core/"
    "heavy_lock.py). The loop keeps its model loaded, so it holds the lock until it ends.",
)
@click.option(
    "--max-failures",
    default=3,
    show_default=True,
    type=click.IntRange(1),
    help="Stop after this many failed experiments in a row.",
)
def loop_command(
    proposer,
    reviewer,
    provider,
    max_experiments,
    epsilon,
    corpus,
    worktree,
    one_heavy_job,
    max_failures,
):
    """Run experiments back to back: draw, claim, audit, review, post and land each one.

    Each runs in a detached worktree reset to origin/main. A model loads once for the
    whole loop. Logs go to logs/loop/, one per experiment, and each prints a RESULT line.
    The loop honours `open-interpretability pause` and the stop file of scripts/experiment-loop.sh.
    """
    import time

    from openinterp import runner_cache
    from openinterp.experiment_loop import Loop, Options
    from openinterp.planner import open_pr_claims, plan_next

    proposer, reviewer = _models(proposer, reviewer, provider)
    repo = Path(_git("rev-parse", "--show-toplevel"))
    worktree = (worktree or repo.parent / f"{repo.name}-loop").resolve()
    corpus = corpus and str(corpus.resolve())
    if not worktree.exists():
        _git("fetch", "--quiet", "origin", "main")
        _git("worktree", "add", "--detach", str(worktree), "origin/main")
    if reviewer == proposer:
        click.echo(
            f"warning: {proposer} both proposes and reviews, so reviews are "
            "self-reviewed; pass another --reviewer for independent ones.",
            err=True,
        )
    if one_heavy_job:  # every model load in this process, and every child, takes the lock
        from openinterp.core.heavy_lock import ENABLE_ENV

        os.environ[ENABLE_ENV] = "1"
    runner_cache.keep()
    os.chdir(worktree)
    options = Options(proposer, reviewer, epsilon, corpus)
    loop = Loop(
        options,
        repo / "logs" / "loop",
        max_experiments,
        max_failures,
        plan_next,
        open_pr_claims,
        time.sleep,
    )
    raise SystemExit(loop.run())


def _models(proposer: str, reviewer: str, provider: str) -> tuple[str, str]:
    """The proposer and reviewer, filled from the provider's profile, whose key it exports."""
    if provider:
        from openinterp import providers

        try:
            profile = providers.load(provider)
        except ValueError as error:
            raise click.UsageError(str(error)) from None
        if not profile.litellm_model and not (proposer and reviewer):
            raise click.UsageError(
                f"{provider} has no LiteLLM model; give --proposer and --reviewer"
            )
        proposer, reviewer = proposer or profile.litellm_model, reviewer or profile.litellm_model
        os.environ.update(providers.litellm_env(profile))
    if not (proposer and reviewer):
        raise click.UsageError("give --proposer and --reviewer, or a --provider")
    return proposer, reviewer


def _git(*args: str) -> str:
    """Run git in the current directory and return its stdout."""
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise click.ClickException(f"git {args[0]} failed: {done.stderr.strip()}")
    return done.stdout.strip()


@cli.command(name="loop-bug")
@click.option("--title", required=True, help="One line: what failed.")
@click.option("--command", "command_", required=True, help="The command that failed.")
@click.option(
    "--error",
    type=click.File("r"),
    default="-",
    show_default=True,
    help="A file with the error output; '-' reads stdin.",
)
@click.option("--branch", default="", help="The branch the loop was on.")
@click.option("--record", default="", help="The record or slug the experiment was about.")
@click.option("--dry-run", is_flag=True, help="Print what would be posted, and post nothing.")
def loop_bug_command(title, command_, error, branch, record, dry_run):
    """File a bug that the loop can't solve, or comment on the open issue of the same bug.

    It removes secrets, email addresses and home-directory paths first. See the
    land-experiment skill for when to use it (openinterp/loop_bugs.py, issue #3261).
    """
    from openinterp.loop_bugs import Report, file

    report = Report(title, command_, error.read(), branch, record)
    try:
        click.echo(file(report, dry_run=dry_run))
    except subprocess.CalledProcessError as failed:
        raise click.ClickException(f"gh failed: {(failed.stderr or '').strip()[-300:]}") from None
