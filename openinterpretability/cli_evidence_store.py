"""``open-interpretability evidence``: move slim records' blobs to the dataset (issue #3254).

``push`` runs where the experiment ran. It pushes the blobs of the records that the branch
adds to ``evidence/<branch>``. A pull request's CI runs ``load``, then ``verify``. After the
merge, the deploy on ``main`` runs ``publish`` with ``HF_EVIDENCE_TOKEN``.
:mod:`openinterpretability.evidence_branch` explains why the blobs ride on their own branch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from openinterpretability.cli import cli, wiki_dir_option

base_ref_option = click.option(
    "--base-ref", default="origin/main", show_default=True, help="The branch the PR merges into."
)
ref_option = click.option(
    "--ref", required=True, help="The evidence branch, for example FETCH_HEAD after a fetch."
)


def _added(wiki_dir: Path, base_ref: str) -> list[Path]:
    """The experiment records added since ``base_ref``: only they have evidence blobs."""
    from openinterpretability.checks import changed_records
    from openinterpretability.reproduce_explanation import is_explanation

    return [p for p in changed_records(wiki_dir, base_ref) if not is_explanation(p)]


@cli.group()
def evidence() -> None:
    """Carry slim records' evidence blobs to the public dataset (docs/analyses/repo-size.md)."""


@evidence.command()
@wiki_dir_option()
@base_ref_option
@click.option("--remote", default="origin", show_default=True)
def push(wiki_dir: Path, base_ref: str, remote: str) -> None:
    """Push the blobs of the records this branch adds to evidence/<branch>."""
    from openinterpretability.evidence_branch import push as push_blobs
    from openinterpretability.evidence_branch import record_refs

    shas = record_refs(_added(wiki_dir, base_ref))
    if not shas:
        click.echo("No slim record on this branch: nothing to push.")
        return
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, check=True
    ).stdout.strip()
    target = push_blobs(branch, shas, remote)
    click.echo(f"Pushed {len(shas)} evidence blobs to {remote}/{target}.")


@evidence.command()
@ref_option
def load(ref: str) -> None:
    """Copy an evidence branch's blobs into the local store, checking each hash."""
    from openinterpretability.evidence_branch import load_branch

    click.echo(f"Loaded {len(load_branch(ref))} evidence blobs from {ref}.")


@evidence.command(name="load-all")
@click.option("--remote", default="origin", show_default=True)
def load_all(remote: str) -> None:
    """Copy the blobs of every evidence branch of ``--remote`` into the local store.

    CI runs it before ``check``. A record that has just merged refers to blobs that the
    deploy may not have published yet, and its evidence branch still holds them.
    """
    from openinterpretability.evidence_branch import LOCAL_REFS, _git, fetch_branches, load_branch

    fetch_branches(remote)
    refs = _git("for-each-ref", "--format=%(refname)", LOCAL_REFS).decode().split()
    count = sum(len(load_branch(ref)) for ref in refs)
    click.echo(f"Loaded {count} evidence blobs from {len(refs)} evidence branches.")


@evidence.command()
@wiki_dir_option()
@click.option("--remote", default="origin", show_default=True)
def publish(wiki_dir: Path, remote: str) -> None:
    """Upload the blobs that this checkout's records need and the dataset lacks.

    Run it on ``main`` after a merge (the deploy does), with a write token. It reads the
    blobs from the remote's evidence branches. It fails if a record's blob is on none.
    """
    from openinterpretability.core.evidence import EVIDENCE_REPO
    from openinterpretability.core.paths import experiments_dir
    from openinterpretability.evidence_branch import fetch_branches, publish_missing

    fetch_branches(remote)
    records = sorted(experiments_dir(wiki_dir).rglob("*.json"))
    uploaded, missing = publish_missing(records)
    click.echo(f"Uploaded {uploaded} evidence blobs to {EVIDENCE_REPO}.")
    if missing:
        raise click.ClickException(f"{len(missing)} blobs are on no evidence branch: {missing}")


@evidence.command()
@wiki_dir_option()
@base_ref_option
def verify(wiki_dir: Path, base_ref: str) -> None:
    """Fail unless each record this branch adds has its evidence, with the right hashes."""
    from openinterpretability.evidence_branch import unresolved

    problems = unresolved(_added(wiki_dir, base_ref))
    for problem in problems:
        click.echo(problem, err=True)
    if problems:
        raise click.ClickException(
            f"{len(problems)} records lack their evidence. Run `open-interpretability evidence "
            "push` on the branch, then run the checks again."
        )
    click.echo("Every added record has its evidence.")
