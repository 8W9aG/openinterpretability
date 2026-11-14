"""``open-interpretability evidence``: move slim records' blobs to the dataset (issue #3254).

``push`` runs where the experiment ran. It pushes the blobs of the records that the branch
adds to ``evidence/<branch>``. CI runs ``upload`` (with ``HF_TOKEN``) or ``load`` (without),
then ``verify``. :mod:`openinterp.evidence_branch` explains why the blobs ride on their own
branch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from openinterp.cli import cli, wiki_dir_option

base_ref_option = click.option(
    "--base-ref", default="origin/main", show_default=True, help="The branch the PR merges into."
)
ref_option = click.option(
    "--ref", required=True, help="The evidence branch, for example FETCH_HEAD after a fetch."
)


def _added(wiki_dir: Path, base_ref: str) -> list[Path]:
    from openinterp.checks import changed_records

    return changed_records(wiki_dir, base_ref)


@cli.group()
def evidence() -> None:
    """Carry slim records' evidence blobs to the public dataset (docs/analyses/repo-size.md)."""


@evidence.command()
@wiki_dir_option()
@base_ref_option
@click.option("--remote", default="origin", show_default=True)
def push(wiki_dir: Path, base_ref: str, remote: str) -> None:
    """Push the blobs of the records this branch adds to evidence/<branch>."""
    from openinterp.evidence_branch import push as push_blobs
    from openinterp.evidence_branch import record_refs

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
    from openinterp.evidence_branch import load_branch

    click.echo(f"Loaded {len(load_branch(ref))} evidence blobs from {ref}.")


@evidence.command()
@ref_option
def upload(ref: str) -> None:
    """Load an evidence branch's blobs, then upload them to the dataset (needs HF_TOKEN)."""
    from openinterp.core.evidence import EVIDENCE_REPO
    from openinterp.evidence_branch import load_branch
    from openinterp.evidence_branch import upload as upload_blobs

    shas = load_branch(ref)
    click.echo(f"Uploaded {upload_blobs(shas)} evidence blobs to {EVIDENCE_REPO}.")


@evidence.command()
@wiki_dir_option()
@base_ref_option
def verify(wiki_dir: Path, base_ref: str) -> None:
    """Fail unless each record this branch adds has its evidence, with the right hashes."""
    from openinterp.evidence_branch import unresolved

    problems = unresolved(_added(wiki_dir, base_ref))
    for problem in problems:
        click.echo(problem, err=True)
    if problems:
        raise click.ClickException(
            f"{len(problems)} records lack their evidence. Run `open-interpretability evidence "
            "push` on the branch, then run the checks again."
        )
    click.echo("Every added record has its evidence.")
