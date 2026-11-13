"""``openinterp build``: compile the archived experiments into the static site."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import cli, wiki_dir_option
from openinterp.sisters import DEFAULT_PATH as SISTERS_PATH


@cli.command()
@wiki_dir_option("Experiment archive (wiki/experiments/**/<id>.json).", exists=True)
@click.option(
    "--output-dir", default=Path("dist"), show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--wiring",
    "wiring_path",
    default=Path("docs/analyses/wiring.jsonl"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Backfilled wiring (`openinterp wiring-backfill`) for records without their own; "
    "skipped if missing.",
)
@click.option(
    "--corpus-sources",
    "source_paths",
    multiple=True,
    default=[Path("corpus.source.json")],
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Corpus sidecars that give the source of records which scanned their corpus "
    "(matched by SHA-256) without recording one, shown as inferred; skipped if missing.",
)
@click.option(
    "--alignment",
    "alignment_path",
    default=Path("docs/analyses/dictionary-alignment.jsonl"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Dictionary latents aligned with each neuron pole (`openinterp crossref`); "
    "skipped if missing.",
)
@click.option(
    "--sisters",
    "sisters_path",
    default=SISTERS_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Each VERIFIED pole's matches in other models (`openinterp sisters`); skipped if missing.",
)
@click.option(
    "--papers",
    "papers_dir",
    default=Path("papers"),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="The papers of the Research section; the section is empty if missing.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=None,
    help="Worker processes that render the unit pages. Default: one per CPU. 1 renders in "
    "this process.",
)
def build(
    wiki_dir, output_dir, wiring_path, source_paths, alignment_path, sisters_path, papers_dir, jobs
):
    """Compile every archived experiment into static pages and data/registry.parquet."""
    from openinterp.builder.compiler import compile_site
    from openinterp.builder.model_facts import load_model_facts
    from openinterp.builder.parallel import default_jobs
    from openinterp.builder.users import load_users
    from openinterp.builder.wiki import load_archive
    from openinterp.core.draws import load_draws, load_stops
    from openinterp.core.errata import load_errata
    from openinterp.core.project import project_links
    from openinterp.corpus import load_sources
    from openinterp.crossref import load_rows as load_alignment
    from openinterp.planner import configured_models
    from openinterp.sisters import load_rows as load_sisters
    from openinterp.wiring_backfill import load_rows

    report = compile_site(
        load_archive(wiki_dir),
        output_dir,
        wiring_rows=load_rows(wiring_path),
        draws=load_draws(wiki_dir),
        corpus_sources=load_sources(source_paths),
        alignment_rows=load_alignment(alignment_path),
        sister_rows=load_sisters(sisters_path),
        papers_dir=papers_dir,
        stops=load_stops(wiki_dir),
        model_facts=load_model_facts(configured_models()),  # config.json only, no weights
        jobs=jobs or default_jobs(),
        errata=load_errata(wiki_dir),
        users=load_users(project_links()["repo_url"], wiki_dir),
    )
    click.echo(
        f"Published {report.experiments_published} experiments: {len(report.pages)} pages, "
        f"{report.parquet} → {output_dir}"
    )
