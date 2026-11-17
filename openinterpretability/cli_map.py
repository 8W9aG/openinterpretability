"""``open-interpretability map <model>``: the concept map of a model's studied neurons (#4422).

It needs the ``map`` extra: ``pip install open-interpretability[map]``. The map goes to
``data/concept-maps/<model slug>.json``, which the site's model page draws.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterpretability.cli import cli, wiki_dir_option


@cli.command(name="map")
@click.argument("model")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Default: data/concept-maps/<model slug>.json.")  # fmt: skip
@wiki_dir_option()
def map_command(model, out, wiki_dir):
    """Place MODEL's studied neuron poles by the meaning of the text they fire on.

    Each pole is embedded from its newest record's top examples, and centred on the corpus
    mean. The poles are then clustered and laid out in two dimensions (concept_map.py).
    """
    from openinterpretability import concept_map as cm

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as missing:
        raise click.ClickException(
            "open-interpretability map needs the map extra: pip install open-interpretability[map]"
        ) from missing
    poles = cm.studied_poles(model, Path(wiki_dir))
    if len(poles) < cm.NEIGHBOURS + 2:
        raise click.ClickException(f"{model} has {len(poles)} studied poles; a map needs more")
    embedder = SentenceTransformer(cm.EMBEDDER, device="cpu")
    found = cm.build(model, poles, lambda texts: embedder.encode(texts, normalize_embeddings=True))
    path = out or cm.map_path(model)
    cm.write(found, path)
    named = sum(c["verified"] for c in found["clusters"])
    click.echo(f"{len(poles)} poles, {len(found['clusters'])} clusters ({named} named from "
               f"VERIFIED members) → {path}")  # fmt: skip
