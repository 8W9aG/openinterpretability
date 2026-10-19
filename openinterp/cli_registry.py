"""Registry-maintenance commands: ``wiki``, ``check`` and ``next``."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import cli


@cli.command()
@click.option(
    "--audits-dir",
    default=Path("audits"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--wiki-dir", default=Path("wiki"), show_default=True, type=click.Path(path_type=Path)
)
def wiki(audits_dir, wiki_dir):
    """Write wiki pages for reviewed records that lack one, and rebuild the indexes."""
    from openinterp.builder.compiler import load_records
    from openinterp.builder.wiki import backfill

    paths = sorted(audits_dir.rglob("*.json"))
    written = backfill(list(zip(load_records(audits_dir), paths, strict=True)), wiki_dir)
    for page in written:
        click.echo(f"Wiki: {page}")
    click.echo(f"{len(written)} new experiment pages; indexes rebuilt in {wiki_dir}")


@cli.command()
@click.option(
    "--audits-dir", default=Path("audits"), show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--wiki-dir", default=Path("wiki"), show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--base-ref",
    help="Git ref to diff against (e.g. origin/main) to enforce write-once wiki pages.",
)
def check(audits_dir, wiki_dir, base_ref):
    """Check that every posted experiment has a valid audit record and wiki page (CI gate)."""
    import os

    from openinterp.checks import check_registry

    report = check_registry(audits_dir, wiki_dir, base_ref=base_ref)
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for problem in report.problems:
        if in_actions:
            click.echo(f"::error file={problem.path}::{problem.message}")
        else:
            click.echo(f"error: {problem}", err=True)

    summary = (
        f"Checked {report.records} audit records and {report.pages} wiki experiment pages: "
        + ("all consistent." if report.ok else f"{len(report.problems)} problem(s).")
    )
    click.echo(summary)
    if in_actions and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(f"### Experiment check\n\n{summary}\n\n")
            fh.writelines(f"- `{p.path}`: {p.message}\n" for p in report.problems)
    if not report.ok:
        raise SystemExit(1)


@cli.command(name="next")
@click.option(
    "--audits-dir", default=Path("audits"), show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--wiki-dir", default=Path("wiki"), show_default=True, type=click.Path(path_type=Path)
)
@click.option("--seed", type=int, help="Reproduce a previous choice. Default: fresh randomness.")
@click.option(
    "--epsilon",
    default=0.35,
    show_default=True,
    type=click.FloatRange(0, 1),
    help="Probability of exploring a uniformly random new neuron.",
)
@click.option(
    "--temperature",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.05),
    help="Higher flattens the weights (more exploration among follow-ups).",
)
@click.option("--model", "models", multiple=True, help="Model(s) to explore. Default: registry's.")
@click.option(
    "--claims/--no-claims",
    default=True,
    show_default=True,
    help="Down-weight neurons claimed by open PRs (needs `gh`).",
)
@click.option("--top", default=0, show_default=True, help="Also list the N heaviest candidates.")
@click.option("--json", "as_json", is_flag=True, help="Print the full plan as JSON.")
def next_experiment(audits_dir, wiki_dir, seed, epsilon, temperature, models, claims, top, as_json):
    """Pick the next experiment (randomised; balances exploration and follow-ups)."""
    from openinterp.planner import open_pr_claims, plan_next

    plan = plan_next(
        audits_dir,
        wiki_dir,
        seed=seed,
        epsilon=epsilon,
        temperature=temperature,
        models=models,
        claims=open_pr_claims() if claims else (),
    )
    if as_json:
        click.echo(plan.to_json())
        return
    c = plan.choice
    click.echo(f"Next experiment: {c.kind}" + (f" on {c.slug}" if c.slug else f" ({c.model})"))
    click.echo(f"Why: {c.rationale}")
    for command in c.commands:
        click.echo(f"  $ {command}")
    if c.skills:
        click.echo(f"Skills: {', '.join(c.skills)}")
    click.echo(
        f"(seed {plan.seed}; {len(plan.candidates)} follow-up candidates; "
        f"{'explored' if plan.explored else 'exploited'})"
    )
    total = sum(x.weight for x in plan.candidates) or 1.0
    for x in plan.candidates[:top]:
        click.echo(f"  {x.weight / total:6.1%}  {x.kind:18s} {x.slug}  — {x.rationale}")
