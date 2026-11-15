"""``open-interpretability corpus``: build a scan corpus from a Hugging Face dataset."""

from __future__ import annotations

from pathlib import Path

import click
from click.core import ParameterSource

from openinterp.blends import BLEND_ALIASES, BLENDS, DEFAULT_DATASET
from openinterp.cli import cli
from openinterp.corpus import LAYOUTS, PRESETS


@cli.command()
@click.option(
    "--dataset",
    default=DEFAULT_DATASET,
    show_default=True,
    help=f"A blend of presets ({', '.join([*BLEND_ALIASES, *BLENDS])}; '{DEFAULT_DATASET}' is "
    f"the latest), a preset ({', '.join(PRESETS)}) or a Hugging Face dataset ID (then pass "
    "--files).",
)
@click.option(
    "--files",
    help="Glob of Parquet files within the dataset; comma-separate several to take an equal "
    "share of documents from each. Default: the preset's.",
)
@click.option(
    "--text-column",
    help="Column holding the text; comma-separate several to join them into one document. "
    "Default: the preset's, or 'text'.",
)
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
    "--detect-lang",
    is_flag=True,
    help="Detect each document's language (fastText lid.176, downloaded once) and record "
    "it as 'lang' where the detector is sure, for datasets that don't say. The source "
    "sidecar records the detector. Replaces a preset's exact languages; not with --lang-pattern.",
)
@click.option(
    "--detect-code",
    is_flag=True,
    help="Detect each document's programming language (Magika) and record it as "
    "'code_language' where the detector is sure, for datasets that don't say (e.g. the "
    "Pile's GitHub files). Prose stays unlabelled. The source sidecar records the "
    "detector. Not with a preset that gives exact programming languages.",
)
@click.option(
    "--chat-template",
    metavar="MODEL",
    help="Write each conversation in --text-column (a list of role/content messages) with "
    "this model's chat template, pinned to its commit. Default: the preset's ("
    + ", ".join(f"{name}: {p.chat_template}" for name, p in PRESETS.items() if p.chat_template)
    + "), or none.",
)
@click.option(
    "--same-as",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Rebuild the corpus this record scanned, at its pinned dataset commits, and check "
    "its SHA-256. It sets the source, so only --out goes with it.",
)
@click.option(
    "--out", default=Path("corpus.jsonl"), show_default=True, type=click.Path(path_type=Path)
)
@click.pass_context
def corpus(
    ctx,
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
    detect_lang,
    detect_code,
    chat_template,
    same_as,
    out,
):
    """Sample a scan corpus from a Hugging Face dataset, without downloading all of it.

    Writes <out> (one {"text": ...} per line) and <name>.source.json next to it, which
    pins the dataset commit, files, seed and licence. `open-interpretability audit --corpus <out>`
    records that source in the submission's provenance.
    """
    from openinterp.corpus import build_from_hub

    if same_as:
        return _same_as(ctx, same_as, out)
    if dataset in BLENDS or dataset in BLEND_ALIASES:
        per_dataset = {
            "--files": files, "--text-column": text_column, "--revision": revision,
            "--layout": layout, "--docs-per-row-group": docs_per_row_group, "--first": first,
            "--lang-pattern": lang_pattern, "--detect-lang": detect_lang,
            "--detect-code": detect_code, "--chat-template": chat_template,
        }  # fmt: skip
        return _blend(dataset, per_dataset, out, documents, seed, max_chars, ascii_)
    preset = PRESETS.get(dataset)
    _check_dataset(dataset, preset, files, lang_pattern, detect_lang, detect_code)
    try:
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
            lang_pattern=None if detect_lang else lang_pattern or (preset and preset.lang_pattern),
            detect_lang=detect_lang,
            code_language_column=preset.code_language_column if preset else None,
            detect_code=detect_code,
            chat_template=chat_template or (preset.chat_template if preset else None),
        )
    except ValueError as error:  # e.g. a model without a chat template
        raise click.ClickException(str(error)) from error
    _report(source, out, documents)


def _check_dataset(dataset, preset, files, lang_pattern, detect_lang, detect_code) -> None:
    """Raise ``click.UsageError`` if the options can't sample ``dataset``."""
    if detect_lang and lang_pattern:
        raise click.UsageError("--detect-lang and --lang-pattern both set the languages.")
    if detect_code and preset and preset.code_language_column:
        raise click.UsageError(f"{dataset!r} has exact programming languages: no --detect-code.")
    if preset is None and not files:
        raise click.UsageError(f"--files is required for {dataset!r}, which is not a preset.")


@cli.command("corpus-cache")
@click.option("--clear", is_flag=True, help="Delete everything cached.")
def corpus_cache(clear):
    """Show the size of (or clear) the cache of Parquet data that `corpus` reads.

    \b
    Location: $OPENINTERP_CORPUS_CACHE (\"off\" disables it), else
    $XDG_CACHE_HOME/openinterp/corpora or ~/.cache/openinterp/corpora.
    Size cap: $OPENINTERP_CORPUS_CACHE_GB (default 5); least recently used go first.
    """
    from openinterp.corpus_cache import ParquetCache

    cache = ParquetCache.default()
    if cache is None:
        click.echo("The corpus cache is off ($OPENINTERP_CORPUS_CACHE).")
        return
    if clear:
        click.echo(f"Deleted {cache.clear()} entries from {cache.root}")
        return
    click.echo(
        f"{cache.root}: {len(cache.entries())} entries, "
        f"{cache.size() / 1e9:.2f} of {cache.max_bytes / 1e9:.0f} GB"
    )


def _same_as(ctx, record: Path, out: Path) -> None:
    from openinterp.cli_review import load_submission
    from openinterp.corpus_rebuild import rebuild, scanned_source

    given = [
        name
        for name in ctx.params
        if name not in ("same_as", "out")
        and ctx.get_parameter_source(name) != ParameterSource.DEFAULT
    ]
    if given:
        raise click.UsageError(f"--same-as sets the source, so not with: {', '.join(given)}.")
    try:
        source = rebuild(scanned_source(load_submission(record)), out)
    except ValueError as error:
        raise click.ClickException(f"{record}: {error}") from error
    _report(source, out, source.documents)


def _blend(dataset, per_dataset: dict, out: Path, documents, seed, max_chars, ascii_) -> None:
    from openinterp.blends import build_blend

    if given := [name for name, value in per_dataset.items() if value]:
        raise click.UsageError(
            f"{', '.join(given)}: set per preset, so not with the blend {dataset!r}."
        )
    source = build_blend(
        dataset, out, documents=documents, seed=seed, max_chars=max_chars, ascii=ascii_
    )
    _report(source, out, documents)
    for part in source.components:
        click.echo(f"  {part.documents} from {part}")


def _report(source, out: Path, documents: int) -> None:
    click.echo(f"Wrote {source.documents} documents to {out}: {source}")
    if source.lang_detector:
        click.echo(f"Languages detected by {source.lang_detector}", err=True)
    if source.code_detector:
        click.echo(f"Programming languages detected by {source.code_detector}", err=True)
    if source.documents < documents:
        click.echo(f"warning: only {source.documents} of {documents} documents found", err=True)
    click.echo(f"Licence: {source.license or 'not stated on the dataset card'}")
