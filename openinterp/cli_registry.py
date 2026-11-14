"""Registry-maintenance commands: ``wiki``, ``stats`` and ``next``."""

from __future__ import annotations

from pathlib import Path

import click
from click.core import ParameterSource

from openinterp.cli import cli, wiki_dir_option
from openinterp.core.paths import experiments_dir
from openinterp.planner import DEFAULT_EPSILON


@cli.command()
@click.option(
    "--import-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Import loose audit records (*.json) found here as experiments.",
)
@wiki_dir_option()
def wiki(import_dir, wiki_dir):
    """Rebuild the wiki indexes, importing loose records as experiments if asked."""
    from openinterp.builder.compiler import load_records
    from openinterp.builder.wiki import backfill

    written = backfill(load_records(import_dir) if import_dir else [], wiki_dir)
    for page in written:
        click.echo(f"Wiki: {page}")
    click.echo(f"{len(written)} new experiments; indexes rebuilt in {wiki_dir}")


def registry_stats(records) -> str:
    """Return the state of the registry.

    That is the experiments, units and models, the VERIFIED and self-reviewed shares, and
    the protocol versions that computed the verdicts.
    """
    from collections import Counter

    def share(n: int, of: int) -> str:
        return f"{n}/{of} ({100 * n / of:.0f}%)" if of else "0/0"

    def summary(rs) -> str:
        units = {r.submission.identifier.slug for r in rs}
        reviewed = [r for r in rs if r.verdict is not None]
        verified = sum(r.is_verified for r in rs)
        self_reviewed = sum(r.self_reviewed for r in reviewed)
        return (
            f"{len(rs)} experiments on {len(units)} units; VERIFIED {share(verified, len(rs))}, "
            f"self-reviewed {share(self_reviewed, len(reviewed))}"
        )

    by_model: dict[str, list] = {}
    for r in records:
        by_model.setdefault(r.submission.identifier.model_name, []).append(r)
    versions = Counter(
        (r.verdict.policy.protocol_version if r.verdict.policy else "< 0.2")
        for r in records
        if r.verdict is not None
    )
    lines = [f"Registry: {summary(records)}, {len(by_model)} models"]
    lines += [f"  {model}: {summary(rs)}" for model, rs in sorted(by_model.items())]
    lines.append(
        "Protocol versions: "
        + (", ".join(f"{v} ×{n}" for v, n in sorted(versions.items(), key=_newest)) or "none")
    )
    return "\n".join(lines)


def _newest(item) -> tuple:
    version = item[0]
    return (1,) if version == "< 0.2" else (0, *(-int(p) for p in version.split(".")))


@cli.command()
@wiki_dir_option()
def stats(wiki_dir):
    """Print the state of the registry.

    It prints experiments, units, models, VERIFIED share, self-reviewed share and the
    protocol versions in use.
    """
    from openinterp.builder.compiler import load_records

    click.echo(registry_stats(load_records(experiments_dir(wiki_dir))))


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--full", is_flag=True, help="Don't cut probes, examples or the Discussion.")
def show(path, full):
    """Summarise a record, submission, scan evidence (ev.json) or a unit's directory.

    Read this instead of opening the JSON. For a record, it shows the verdict, failed
    gates and probes, scores and each page's Discussion. For a scan, it shows the corpus,
    peak and top examples.
    """
    from openinterp.show import show as summarise

    try:
        click.echo(summarise(path, full))
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="PATH") from exc


def _refuse_while_paused() -> None:
    """Exit 75 with the reason if `open-interpretability pause` is in force."""
    from openinterp.core import pause

    if paused := pause.read():
        click.echo(paused.describe(), err=True)
        raise SystemExit(pause.EXIT_PAUSED)


@cli.command(name="next")
@wiki_dir_option()
@click.option("--seed", type=int, help="Reproduce a previous choice. Default: fresh randomness.")
@click.option(
    "--epsilon",
    default=DEFAULT_EPSILON,
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
@click.option(
    "--model",
    "models",
    multiple=True,
    help="Model(s) to explore. Default: [tool.openinterp] models in pyproject.toml.",
)
@click.option(
    "--claims/--no-claims",
    default=True,
    show_default=True,
    help="Down-weight neurons claimed by open PRs (needs `gh`).",
)
@click.option("--top", default=0, show_default=True, help="Also list the N heaviest candidates.")
@click.option("--json", "as_json", is_flag=True, help="Print the full plan as JSON.")
@click.option(
    "--steps",
    is_flag=True,
    help="Also print every step of the draw as exact commands: claim, history, run, post.",
)
@click.option(
    "--corpus",
    default="corpus.jsonl",
    show_default=True,
    help="For --steps: the corpus a new scan reads.",
)
@click.option(
    "--fresh-corpus",
    is_flag=True,
    help="For --steps: a new scan first builds its own corpus (the default blend, with the "
    "draw's seed). Not with --corpus.",
)
@click.option(
    "--proposer",
    help="For --steps: a LiteLLM model for `audit`. Default: the neuron-proposer subagent.",
)
@click.option(
    "--reviewer",
    help="For --steps: a LiteLLM model for `verify`. Default: the neuron-reviewer and "
    "explanation-judge subagents.",
)
def next_experiment(
    wiki_dir,
    seed,
    epsilon,
    temperature,
    models,
    claims,
    top,
    as_json,
    steps,
    corpus,
    fresh_corpus,
    proposer,
    reviewer,
):
    """Pick the next experiment (randomised; balances exploration and follow-ups).

    While `open-interpretability pause` is in force it draws nothing and exits 75.
    """
    from openinterp.planner import open_pr_claims, plan_next

    _refuse_while_paused()

    source = click.get_current_context().get_parameter_source("corpus")
    if fresh_corpus and source != ParameterSource.DEFAULT:
        raise click.UsageError("--fresh-corpus and --corpus both set the scan's corpus.")
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
    if any(
        cmd.startswith("open-interpretability audit") and "--neuron" in cmd for cmd in c.commands
    ):
        _scan_cache_note(c)
    if c.skills:
        click.echo(f"Skills: {', '.join(c.skills)}")
    click.echo(
        f"(seed {plan.seed}; {len(plan.candidates)} follow-up candidates; "
        f"{'explored' if plan.explored else 'exploited'})"
    )
    _list_candidates(plan, top)
    if steps:
        _print_steps(plan, wiki_dir, corpus, proposer, reviewer, fresh_corpus)


def _print_steps(plan, wiki_dir, corpus, proposer, reviewer, fresh_corpus) -> None:
    """Record the draw in the ledger, then print its steps, whose claim commits the entry."""
    from openinterp.draw_steps import render
    from openinterp.draw_steps import steps as draw_steps

    ledger = record_draw_entry(plan, wiki_dir)
    click.echo(f"\nDraw recorded in the ledger: {ledger}")
    click.echo("\nSteps:")
    click.echo(render(draw_steps(plan, corpus, proposer, reviewer, fresh_corpus, ledger)))


@cli.command(name="ledger")
@wiki_dir_option()
@click.option("--json", "as_json", is_flag=True, help="Print the summary as JSON.")
def ledger_report(wiki_dir, as_json):
    """Report every recorded draw's outcome, and VERIFIED rates against all draws.

    The ledger (wiki/ledger/) holds every draw that `open-interpretability next --steps` made,
    before its run. A draw with no record and no stop was dropped, or is still running.
    """
    import json

    from openinterp.core.ledger import load_ledger, resolve, summary

    entries = load_ledger(Path(wiki_dir))
    records, stopped = _ledger_rows(Path(wiki_dir))
    resolved = [(e, *resolve(e, records, stopped)) for e in entries]
    result = summary(resolved)
    if as_json:
        click.echo(json.dumps(result, indent=1))
        return
    for e, outcome, status in resolved:
        what = e.slug or f"{e.model} (seed {e.audit_seed})"
        click.echo(f"{e.timestamp:%Y-%m-%d %H:%M} {e.kind:<18} {what}: {status or outcome}")
    n = result["explore_entries"]
    click.echo(
        f"{result['entries']} draws: {result['outcomes']}. Explore draws: {n}, "
        f"{result['explore_verified']} VERIFIED."
    )
    if n:
        share = result["verified_share_of_finished_explore"]
        click.echo(
            f"VERIFIED share of explore draws: {result['verified_share_of_all_explore']:.1%} "
            f"of all, {share:.1%} of finished."
            if share is not None
            else ""
        )


def _ledger_rows(wiki_dir: Path):
    """(slug, random seed, time, status) of every record, and of every draw or stop."""
    from openinterp.core.draws import load_draws, load_stops
    from openinterp.core.schemas import AuditRecord

    records = []
    for path in sorted(experiments_dir(wiki_dir).rglob("*.json")):
        r = AuditRecord.model_validate_json(path.read_text())
        if r.verdict is None:
            continue
        p = r.submission.provenance
        seed = p.seed if p and p.selection == "random" else None
        records.append((r.submission.identifier.slug, seed, r.verdict.timestamp, r.verdict.status))
    stopped = [
        (d.identifier.slug, d.provenance.seed, d.timestamp, None)
        for d in [*load_draws(wiki_dir), *load_stops(wiki_dir)]
    ]
    return records, stopped


def record_draw_entry(plan, wiki_dir) -> str:
    """Write the draw's ledger entry before any run (issue #3205), and return its path."""
    from datetime import UTC, datetime

    from openinterp.core.ledger import DrawEntry, write_entry
    from openinterp.draw_steps import draw_slug

    c = plan.choice
    options = (c.audit or "").split()
    entry = DrawEntry(
        timestamp=datetime.now(UTC).replace(microsecond=0),
        seed=plan.seed,
        kind=c.kind,
        model=c.model,
        slug=draw_slug(c),
        record=c.record,
        audit=c.audit,
        pole_forced=("--pole" in options) if "--random" in options else None,
    )
    return str(write_entry(entry, Path(wiki_dir)))


def _scan_cache_note(c) -> None:
    from openinterp.harness.layer_cache import LayerCache

    cache = LayerCache.default()
    if cache is not None and c.layer in cache.cached_layers(c.model):
        click.echo(
            f"Scan cache: layer {c.layer} is cached, so the audit replays its scan "
            "(when the corpus options match) instead of running the model."
        )


def _list_candidates(plan, top: int) -> None:
    """The ``top`` heaviest follow-ups by share of the weight, and the explore draws."""
    total = sum(x.weight for x in plan.candidates) or 1.0
    for x in plan.candidates[:top]:
        click.echo(f"  {x.weight / total:6.1%}  {x.kind:18s} {x.slug}  — {x.rationale}")
    if top and len(plan.explore_models) > 1:
        click.echo("Explore draws by model (neurons covered):")
        for model, p in sorted(plan.explore_models.items(), key=lambda kv: -kv[1]):
            focus = "  ← smallest model below the verified target" if model == plan.focus else ""
            click.echo(f"  {p:6.1%}  {model}  {plan.coverage[model]}{focus}")
