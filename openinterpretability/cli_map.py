"""``open-interpretability map``: one concept map of every model's studied neurons (#4422, #4586).

It needs the ``map`` extra: ``pip install open-interpretability[map]``. The map goes to
``data/concept-maps/shared.json``. Each model page draws it, and so does the atlas page.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterpretability.cli import cli, wiki_dir_option


@cli.command(name="map")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Default: data/concept-maps/shared.json.")  # fmt: skip
@click.option("--refit", is_flag=True,
              help="Fit the map again, from the old positions. Old points move.")  # fmt: skip
@wiki_dir_option()
def map_command(out, refit, wiki_dir):
    """Place every configured model's studied poles by the meaning of the text they fire on.

    Each pole is embedded from its newest record's top examples, and centred on the pooled
    mean. A first run fits the map. A later run keeps each old pole where it is and places
    the new ones among their nearest neighbours (concept_map.py).
    """
    from openinterpretability import concept_map as cm
    from openinterpretability.planner import configured_models

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as missing:
        raise click.ClickException(
            "open-interpretability map needs the map extra: pip install open-interpretability[map]"
        ) from missing
    path = out or cm.SHARED
    poles = cm.studied_poles(set(configured_models()), Path(wiki_dir))
    if len(poles) < cm.NEIGHBOURS + 2:
        raise click.ClickException(f"{len(poles)} studied poles; a map needs more")
    old = cm.load(path)
    embedder = SentenceTransformer(cm.EMBEDDER, device="cpu")
    found = cm.build(poles, lambda texts: embedder.encode(texts, normalize_embeddings=True),
                     old, refit=refit)  # fmt: skip
    cm.write(found, path)
    added = found["poles"] - (len(old["points"]) if old else 0)
    named = sum(c["verified"] for c in found["clusters"])
    click.echo(f"{found['poles']} poles ({added} new), {len(found['clusters'])} regions "
               f"({named} named), version {found['version']} → {path}")  # fmt: skip
