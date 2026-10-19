"""Registry-maintenance commands: ``wiki``, ``check`` and ``next``."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import cli


@cli.command()
@click.option(
    "--import-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Import loose audit records (*.json) found here as experiments.",
)
@click.option(
    "--wiki-dir", default=Path("wiki"), show_default=True, type=click.Path(path_type=Path)
)
def wiki(import_dir, wiki_dir):
    """Rebuild the wiki indexes, importing loose records as experiments if asked."""
    from openinterp.builder.compiler import load_records
    from openinterp.builder.wiki import backfill

    written = backfill(load_records(import_dir) if import_dir else [], wiki_dir)
    for page in written:
        click.echo(f"Wiki: {page}")
    click.echo(f"{len(written)} new experiments; indexes rebuilt in {wiki_dir}")


@cli.command()
@click.option(
    "--wiki-dir", default=Path("wiki"), show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--base-ref",
    help="Git ref to diff against (e.g. origin/main) to enforce write-once wiki pages.",
)
def check(wiki_dir, base_ref):
    """Check that every posted experiment has a valid record and wiki page (CI gate)."""
    import os

    from openinterp.checks import check_registry

    report = check_registry(wiki_dir, base_ref=base_ref)
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for problem in report.problems:
        if in_actions:
            click.echo(f"::error file={problem.path}::{problem.message}")
        else:
            click.echo(f"error: {problem}", err=True)

    summary = (
        f"Checked {report.records} experiment records and {report.pages} experiment pages: "
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
def next_experiment(wiki_dir, seed, epsilon, temperature, models, claims, top, as_json):
    """Pick the next experiment (randomised; balances exploration and follow-ups)."""
    from openinterp.planner import open_pr_claims, plan_next

    plan = plan_next(
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


@cli.command()
@click.argument("records", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--base-ref",
    help="Reproduce every record added or changed since this git ref (e.g. origin/main).",
)
@click.option(
    "--wiki-dir", default=Path("wiki"), show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--atol",
    default=1e-3,
    show_default=True,
    type=click.FloatRange(min=0),
    help="Absolute tolerance, as a fraction of the neuron's max activation.",
)
@click.option("--rtol", default=1e-2, show_default=True, type=click.FloatRange(min=0))
@click.option("--device", default="auto", show_default=True, help="cpu, cuda, mps or auto.")
def reproduce(records, base_ref, wiki_dir, atol, rtol, device):
    """Re-run experiment records against their pinned model and compare (no LLM calls).

    Each record's scan evidence, counterfactual deltas and review are measured again
    with the model revision and dtype it pins, under its own review policy. Exits 1 if
    any number differs beyond tolerance or any pass/fail, gate or status differs.
    """
    from openinterp.checks import changed_records
    from openinterp.cli import _load_runner
    from openinterp.core.schemas import AuditRecord
    from openinterp.reproduce import Tolerance, check_protocol, pinned_environment
    from openinterp.reproduce import reproduce_record as run

    paths = list(records)
    if base_ref:
        paths += [p for p in changed_records(wiki_dir, base_ref) if p not in paths]
    if not paths:
        click.echo("No records to reproduce.")
        return
    tolerance = Tolerance(atol=atol, rtol=rtol)
    runners: dict[tuple, object] = {}
    failed = 0
    for path in paths:
        record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
        reason = check_protocol(record)
        if reason is None:
            revision, dtype = pinned_environment(record)
            key = (record.submission.identifier.model_name, revision, dtype)
            if key not in runners:
                runners[key] = _load_runner(key[0], device, dtype, revision=revision)
            report = run(record, runners[key], tolerance, path)
            problems, warnings = report.mismatches, report.warnings
        else:
            problems, warnings = [reason], []
        failed += bool(problems)
        click.echo(f"{'FAILED' if problems else 'REPRODUCED'}: {path}")
        for message in problems:
            click.echo(f"  mismatch: {message}")
        for message in warnings:
            click.echo(f"  warning: {message}")
    click.echo(f"{len(paths) - failed}/{len(paths)} record(s) reproduced.")
    if failed:
        raise SystemExit(1)
