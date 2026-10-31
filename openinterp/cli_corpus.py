"""``openinterp corpus``: build a scan corpus from a Hugging Face dataset."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import cli
from openinterp.corpus import LAYOUTS, PRESETS


@cli.command()
@click.option(
    "--dataset",
    default="fineweb-edu",
    show_default=True,
    help=f"A preset ({', '.join(PRESETS)}) or a Hugging Face dataset ID (then pass --files).",
)
@click.option(
    "--files",
    help="Glob of Parquet files within the dataset; comma-separate several to take an equal "
    "share of documents from each. Default: the preset's.",
)
@click.option("--text-column", help="Column holding the text. Default: the preset's, or 'text'.")
@click.option(
    "--revision",
    help="Dataset branch, tag or commit. Default: the preset's, or main; pinned on use.",
)
@click.option(
    "--layout",
    type=click.Choice(list(LAYOUTS)),
    help="Join rows into documents this way, e.g. one WikiText row per paragraph. Default: "
    "the preset's, or one document per row.",
)
@click.option("--docs", "documents", default=500, show_default=True, type=click.IntRange(1))
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--max-chars",
    default=3000,
    show_default=True,
    type=click.IntRange(1),
    help="Cut each document to this many characters.",
)
@click.option(
    "--docs-per-row-group",
    type=click.IntRange(1),
    help="Documents taken from each Parquet row group read; lower spreads the sample wider. "
    "Default: the preset's, or 20.",
)
@click.option(
    "--first",
    is_flag=True,
    help="Take the dataset's first --docs documents in file order, cut but not stripped, "
    "instead of a seeded sample.",
)
@click.option(
    "--ascii",
    "ascii_",
    is_flag=True,
    help="Escape non-ASCII characters in the JSONL. With --first, rebuilds the registry's "
    "corpus from pile-10k (README).",
)
@click.option(
    "--lang-pattern",
    help="Regular expression whose group, matched in each --files glob, is the language its "
    "documents record as 'lang' (e.g. '^\\d+\\.(\\w+)/' for Wikipedia). Default: the preset's.",
)
@click.option(
    "--out", default=Path("corpus.jsonl"), show_default=True, type=click.Path(path_type=Path)
)
def corpus(
    dataset,
    files,
    text_column,
    revision,
    layout,
    documents,
    seed,
    max_chars,
    docs_per_row_group,
    first,
    ascii_,
    lang_pattern,
    out,
):
    """Sample a scan corpus from a Hugging Face dataset, without downloading all of it.

    Writes <out> (one {"text": ...} per line) and <name>.source.json next to it, which
    pins the dataset commit, files, seed and licence. `openinterp audit --corpus <out>`
    records that source in the submission's provenance.
    """
    from openinterp.corpus import build_from_hub

    preset = PRESETS.get(dataset)
    if preset is None and not files:
        raise click.UsageError(f"--files is required for {dataset!r}, which is not a preset.")
    source = build_from_hub(
        preset.dataset if preset else dataset,
        out,
        files=files or preset.files,
        text_column=text_column or (preset.text_column if preset else "text"),
        revision=revision or (preset.revision if preset else None),
        layout=layout or (preset.layout if preset else None),
        documents=documents,
        seed=seed,
        max_chars=max_chars,
        docs_per_row_group=docs_per_row_group or (preset.docs_per_row_group if preset else 20),
        first=first,
        ascii=ascii_,
        lang_pattern=lang_pattern or (preset.lang_pattern if preset else None),
    )
    click.echo(f"Wrote {source.documents} documents to {out}: {source}")
    if source.documents < documents:
        click.echo(f"warning: only {source.documents} of {documents} documents found", err=True)
    click.echo(f"Licence: {source.license or 'not stated on the dataset card'}")
