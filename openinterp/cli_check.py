"""The ``check`` command: registry consistency, for CI and before a push."""

from __future__ import annotations

import click

from openinterp.cli import cli, wiki_dir_option


@cli.command()
@wiki_dir_option()
@click.option(
    "--base-ref",
    help="Git ref to diff against (e.g. origin/main). Enforces write-once pages, and checks "
    "only the branch's changes unless it also changes files outside the wiki.",
)
def check(wiki_dir, base_ref):
    """Check that every posted experiment has a valid record and wiki page (CI gate)."""
    import os
    import subprocess

    from openinterp.checks import check_registry

    try:
        report = check_registry(wiki_dir, base_ref=base_ref)
    except subprocess.CalledProcessError as error:
        raise click.ClickException(
            f"can't diff against {base_ref} ({error.stderr.strip()}). "
            "Fetch the ref, or run `open-interpretability check` to check every record."
        ) from error
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for problem in report.problems:
        if in_actions:
            click.echo(f"::error file={problem.path}::{problem.message}")
        else:
            click.echo(f"error: {problem}", err=True)

    summary = (
        f"Checked {report.records} experiment records and {report.pages} experiment pages"
        + "".join(f" and {n} {kind} records" for kind, n in report.extra_counts.items() if n)
        + (f" changed since {base_ref}" if report.changed_only else "")
        + ": "
        + ("all consistent." if report.ok else f"{len(report.problems)} problem(s).")
    )
    click.echo(summary)
    if in_actions and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(f"### Experiment check\n\n{summary}\n\n")
            fh.writelines(f"- `{p.path}`: {p.message}\n" for p in report.problems)
    if not report.ok:
        raise SystemExit(1)
