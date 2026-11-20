"""``open-interpretability build``: compile the archived experiments into the static site."""

from __future__ import annotations

from pathlib import Path

import click

from openinterpretability.builder.pending import open_experiments
from openinterpretability.cli import cli, wiki_dir_option
from openinterpretability.core.schemas import COMMITTED_SOURCES
from openinterpretability.sisters import DEFAULT_PATH as SISTERS_PATH


def _wiki_inputs(wiki_dir, states: dict) -> dict:
    """What the build reads from the wiki besides the experiments, by compile_site argument.

    ``states`` (``planner_states.states_of``) come from the records that the build loaded.
    """
    from openinterpretability.builder.circuit_units import load_circuit_data
    from openinterpretability.builder.concept_map_view import load_maps
    from openinterpretability.builder.explanation_pages import load_views
    from openinterpretability.concept_map import MAPS_DIR
    from openinterpretability.core.draws import load_draws, load_stops
    from openinterpretability.core.errata import load_errata

    return {
        "draws": load_draws(wiki_dir),  # random draws no experiment could audit
        "stops": load_stops(wiki_dir),
        "errata": load_errata(wiki_dir),
        "explanations": load_views(wiki_dir, states),  # each prompt's explanations, and if stale
        "circuits": load_circuit_data(wiki_dir),  # registry circuits, their records, hub norms
        "concept_map": load_maps(MAPS_DIR),  # the shared concept map (#4422, #4586)
    }


def _load(wiki_dir: Path):
    """Return the slim archive, its records and the other inputs, as the build holds them.

    Each record is held slim as it loads (``builder.slim.SlimArchive``, #4652). The records
    are the one copy that the build holds: ``compile_site`` takes them over.
    """
    from openinterpretability.builder.slim import SlimArchive
    from openinterpretability.builder.wiki import load_archive_paths
    from openinterpretability.core.errata import load_errata
    from openinterpretability.planner_states import states_of

    archive = SlimArchive(load_errata(wiki_dir))
    pairs = load_archive_paths(wiki_dir, keep=archive)
    inputs = _wiki_inputs(wiki_dir, states_of(pairs, archive.errata))
    return archive, [record for _, record in pairs], inputs


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
    help="Backfilled wiring (`open-interpretability wiring-backfill`) for records without "
    "their own; skipped if missing.",
)
@click.option(
    "--corpus-sources",
    "source_paths",
    multiple=True,
    default=[Path(name) for name in COMMITTED_SOURCES],
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
    help="Dictionary latents aligned with each neuron pole (`open-interpretability crossref`); "
    "skipped if missing.",
)
@click.option(
    "--sisters",
    "sisters_path",
    default=SISTERS_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Each VERIFIED pole's matches in other models (`open-interpretability sisters`); "
    "skipped if missing.",
)
@click.option(
    "--coverage",
    "coverage_path",
    default=Path("docs/analyses/registry-coverage.jsonl"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Registry coverage rows (`open-interpretability coverage`) for the home page; "
    "skipped if missing.",
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
    "--paper-pdfs/--no-paper-pdfs",
    default=True,
    show_default=True,
    help="Print a PDF of each paper with headless Chrome ($OPENINTERPRETABILITY_CHROME, else the "
    "usual paths). Skipped, with a note, when no Chrome is found.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=None,
    help="Worker processes that render the unit pages. Default: one per CPU. 1 renders in "
    "this process.",
)
def build(  # noqa: PLR0913 (one option per input file)
    wiki_dir,
    output_dir,
    wiring_path,
    source_paths,
    alignment_path,
    sisters_path,
    coverage_path,
    papers_dir,
    paper_pdfs,
    jobs,
):
    """Compile every archived experiment into static pages and data/registry.parquet."""
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.coverage import load_rows as load_coverage
    from openinterpretability.builder.model_facts import load_model_facts
    from openinterpretability.builder.parallel import default_jobs
    from openinterpretability.builder.users import load_users
    from openinterpretability.core.project import project_links
    from openinterpretability.core.takedowns import activate, load_takedowns
    from openinterpretability.corpus import load_sources
    from openinterpretability.crossref import load_rows as load_alignment
    from openinterpretability.planner import configured_models
    from openinterpretability.sisters import load_rows as load_sisters
    from openinterpretability.wiring_backfill import load_rows

    activate(load_takedowns(wiki_dir))  # before any text is scrubbed; workers fork after this
    archive, records, inputs = _load(wiki_dir)
    report = compile_site(
        records,
        output_dir,
        wiring_rows=load_rows(wiring_path),
        corpus_sources=load_sources(source_paths),
        alignment_rows=load_alignment(alignment_path),
        sister_rows=load_sisters(sisters_path),
        papers_dir=papers_dir,
        model_facts=load_model_facts(configured_models()),  # config.json only, no weights
        jobs=jobs or default_jobs(),
        users=load_users(project_links()["repo_url"], wiki_dir),
        coverage_rows=load_coverage(coverage_path),
        paper_pdfs=paper_pdfs,
        pending=open_experiments(),
        archive=archive,
        **inputs,
    )
    click.echo(f"Published {report.experiments_published} experiments: {len(report.pages)} "
               f"pages, {report.parquet} → {output_dir}")  # fmt: skip
