"""``openinterp sisters``: VERIFIED neuron poles matched to other models' poles.

See :mod:`openinterp.sisters`.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import _load_runner, cli, wiki_dir_option
from openinterp.sisters import DEFAULT_PATH


@cli.command("sisters")
@wiki_dir_option()
@click.option(
    "--model",
    "model_name",
    default=None,
    help="Match only this model's VERIFIED poles. Default: every configured model "
    "([tool.openinterp] models in pyproject.toml).",
)
@click.option(
    "--other",
    "others",
    multiple=True,
    help="Search only this model (repeatable). Default: every other configured model, "
    "and the model's larger siblings (openinterp.sisters.SIBLINGS).",
)
@click.option(
    "--corpus",
    default=Path("corpus.jsonl"),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Texts both models read; each document is one forward pass.",
)
@click.option(
    "--out",
    "path",
    default=DEFAULT_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="JSONL of sister rows, one per pole and other model.",
)
@click.option("-k", "k", default=5, show_default=True, help="Poles kept per pole and model.")
@click.option("--device", default="auto", show_default=True, help="cpu, cuda, mps or auto.")
def sisters_command(wiki_dir, model_name, others, corpus, path, k, device):
    """Find each VERIFIED pole's best-correlated poles in other models.

    Both models read --corpus. They are compared where both have read the same text
    (token ends they share), so they may tokenize differently. Every pole of every layer
    of the other model is correlated with each VERIFIED pole. Then each match is
    correlated with every pole of the first model, so a row says whether the match is
    mutual. Poles already in --out for that
    model are skipped, so a run can resume.
    """
    from openinterp.builder.wiki import load_archive
    from openinterp.evidence import read_corpus
    from openinterp.planner import configured_models
    from openinterp.sisters import pairs, sisters, sort_rows

    todo = pairs(configured_models(wiki_dir.parent / "pyproject.toml"))
    if model_name:
        todo = [p for p in todo if p[0] == model_name] or pairs([model_name])
    if others:
        todo = [(a, b) for a in {a for a, _ in todo} or {model_name} for b in others if a != b]
    if not todo:
        raise click.UsageError("no model pairs to match; pass --model and --other")
    texts, corpus_name, corpus_sha, _ = read_corpus(corpus)
    records = load_archive(wiki_dir)

    def load(name, revision):  # not cached: each stage frees its model before the next
        return _load_runner(name, device, "float32", revision)

    wrote = sisters(
        records, load, path, texts, todo, corpus=corpus_name, corpus_sha256=corpus_sha, k=k,
        echo=click.echo,
    )  # fmt: skip
    for (source, other), n in wrote.items():
        click.echo(f"Matched {n} poles of {source} against {other} → {path}")
    if path.exists():
        sort_rows(path)
