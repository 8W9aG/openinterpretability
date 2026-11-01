"""``openinterp concepts``: a labelled concept set from an annotated dataset.

See :mod:`openinterp.concepts` for the presets and how labels become spans.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import cli
from openinterp.concepts import CONCEPT_PRESETS


@cli.command()
@click.option("--dataset", "preset", required=True, type=click.Choice(list(CONCEPT_PRESETS)))
@click.option(
    "--label",
    help="The label to mark (PER, NOUN, nsubj, fr, Sports). Omit it to list the split's "
    "labels and how many words or documents carry each.",
)
@click.option("--split", help="Dataset split. Default: the preset's held-out one.")
@click.option("--subset", help="Universal Dependencies treebank (default en_ewt).")
@click.option("--column", help="Read labels from this column instead (e.g. UD's xpos).")
@click.option("--docs", "documents", default=1000, show_default=True, type=click.IntRange(1))
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--max-chars",
    default=3000,
    show_default=True,
    type=click.IntRange(1),
    help="Join sentences into documents of at most this many characters.",
)
@click.option("--out", type=click.Path(path_type=Path), help="Concept set JSONL.")
def concepts(preset, label, split, subset, column, documents, seed, max_chars, out):
    """Write a concept set from human-annotated labels: JSON lines
    {"text": ..., "spans": [[start, end], ...]} for `openinterp probe-compare --train/--test`,
    sparse-probing searches and reviewers' positive holdouts.

    Word tags (conll2003-*, ud-*) mark each tagged word; document classes
    (language-identification, ag_news) mark whole documents, half of them of the class.
    Also writes <name>.source.json, which pins the dataset commit, split and licence.
    """
    from openinterp.concepts import build_concept, load_split

    if label is None:
        loaded = load_split(preset, split=split, subset=subset, column=column, max_chars=max_chars)
        click.echo(
            f"{loaded.dataset}@{loaded.revision[:8]} {loaded.files}: {len(loaded.docs)} docs"
        )
        for tag, n in loaded.counts().most_common():
            click.echo(f"{n:>8,}  {tag}")
        return
    if out is None:
        raise click.UsageError("--out is required with --label.")
    try:
        source = build_concept(
            preset,
            label,
            out,
            split=split,
            subset=subset,
            column=column,
            documents=documents,
            seed=seed,
            max_chars=max_chars,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote {out}: {source}")
    if source.skipped_sentences:
        click.echo(f"warning: {source.skipped_sentences} sentences' words not found", err=True)
    click.echo(f"Licence: {source.license or 'not stated on the dataset card'}")
