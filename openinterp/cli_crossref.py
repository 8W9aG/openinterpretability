"""``openinterp crossref`` and ``openinterp aligned-latents``: neuron poles against
published dictionaries. See :mod:`openinterp.crossref`."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import _load_runner, cli, wiki_dir_option
from openinterp.crossref import DEFAULT_PATH

out_option = click.option(
    "--out",
    "path",
    default=DEFAULT_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="JSONL of alignment rows, one per pole and dictionary.",
)


@cli.command("crossref")
@wiki_dir_option()
@out_option
@click.option("-k", "k", default=5, show_default=True, help="Latents kept per side.")
@click.option("--device", default="auto", show_default=True, help="cpu, cuda, mps or auto.")
def crossref_command(wiki_dir, path, k, device):
    """Find the dictionary latents aligned with every archived neuron pole's weights.

    Covers each pole whose model has a dictionary preset at its layer; rows already in
    --out are skipped, so a run can resume. Weights only: no text is run.
    """
    from openinterp.builder.wiki import load_archive
    from openinterp.crossref import crossref, sort_rows

    def load(model_name, revision):
        return _load_runner(model_name, device, "float32", revision)

    wrote = crossref(load_archive(wiki_dir), load, path, k=k, echo=click.echo)
    if path.exists():
        sort_rows(path)
    click.echo(f"Aligned {wrote} pole–dictionary pairs → {path}")


@cli.command("aligned-latents")
@click.argument("slug")
@out_option
@wiki_dir_option()
def aligned_latents(slug, path, wiki_dir):
    """Show a pole's aligned latents (e.g. Qwen--Qwen2.5-0.5B/L3/N12-positive) with their
    descriptions, from this registry or Neuronpedia."""
    from openinterp.builder.wiki import load_archive
    from openinterp.crossref import describe, load_rows, neuronpedia_url, registry_explanations

    rows = load_rows(path).get(slug)
    if not rows:
        raise click.ClickException(f"no alignment rows for {slug} in {path}; run crossref")
    registry = registry_explanations(load_archive(wiki_dir))
    described = {
        (id(d["row"]), d["side"], d["match"].feature_idx): d for d in describe(rows, registry)
    }
    for row in rows:
        for side_name in ("writes", "reads"):
            side = getattr(row, side_name)
            if side is None:
                continue
            click.echo(
                f"{row.dictionary} {side_name} (chance {side.chance:.2f}, "
                f"best beats {side.percentile:.0%} of the layer's neurons)"
            )
            for m in side.matches:
                d = described.get((id(row), side_name, m.feature_idx))
                channel = f" {m.channel}" if m.channel and side_name == "reads" else ""
                text = (d and d["description"]) or ("below chance" if d is None else "—")
                url = neuronpedia_url(row, m.feature_idx)
                click.echo(f"  F{m.feature_idx:<6} {m.cosine:+.3f}{channel}  {text}")
                if url:
                    click.echo(f"          {url}")
