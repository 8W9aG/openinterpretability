"""``openinterp loop``: run experiments back to back from the CLI, with no Claude session."""

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
@click.option("--proposer", required=True, callback=_api_model, help="LiteLLM proposer model.")
@click.option(
    "--reviewer",
    required=True,
    callback=_api_model,
    help="LiteLLM reviewer model. A review draw of this model's own record is drawn again.",
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
    help="Exploration probability, as in `openinterp next`.",
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
    proposer, reviewer, max_experiments, epsilon, corpus, worktree, one_heavy_job, max_failures
):
    """Run experiments back to back: draw, claim, audit, review, post and land each one.

    Each runs in a detached worktree reset to origin/main. A model loads once for the
    whole loop. Logs go to logs/loop/, one per experiment, and each prints a RESULT line.
    The loop honours `openinterp pause` and the stop file of scripts/experiment-loop.sh.
    """
    import time

    from openinterp import runner_cache
    from openinterp.experiment_loop import Loop, Options
    from openinterp.planner import open_pr_claims, plan_next

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


def _git(*args: str) -> str:
    """Run git in the current directory and return its stdout."""
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise click.ClickException(f"git {args[0]} failed: {done.stderr.strip()}")
    return done.stdout.strip()
