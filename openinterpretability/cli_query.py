"""``open-interpretability query``: search the registry without cloning it (#3242).

See :mod:`openinterpretability.registry_query`.
"""

from __future__ import annotations

import json

import click

from openinterpretability.cli import cli
from openinterpretability.registry_query import MAX_LIMIT, search, unit_label


@cli.command("query")
@click.option(
    "--source",
    default=None,
    help="Path or URL of registry.parquet. Default: dist/data/registry.parquet if it "
    "exists, else the published one on the site.",
)
@click.option("--model", default=None, help="Model name, such as openai-community/gpt2.")
@click.option("--layer", type=int, default=None, help="Layer.")
@click.option("--kind", default=None, help="Unit kind, such as mlp_neuron_pole or attn_head.")
@click.option("--status", default=None, help="The unit's standing, such as VERIFIED.")
@click.option("--text", default=None, help="Text that the explanation contains (any case).")
@click.option("--token", default=None, help="A token that the unit fires on, exactly.")
@click.option("--all", "all_records", is_flag=True,
              help="Every experiment, not only each unit's newest.")  # fmt: skip
@click.option("--limit", type=click.IntRange(1, MAX_LIMIT), default=20, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print JSON, one object per line.")
def _query(source, model, layer, kind, status, text, token, all_records, limit, as_json):
    """Search the registry's experiments by model, layer, kind, standing, text or token.

    \f
    Args:
        source: Path or URL of registry.parquet, or None for the default.
        model: Model name.
        layer: Layer.
        kind: Unit kind.
        status: The unit's standing.
        text: Text that the explanation contains.
        token: A token that the unit fires on.
        all_records: Every experiment, not only each unit's newest.
        limit: The most rows.
        as_json: Print JSON lines, not a table.
    """
    rows = search(source, model=model, layer=layer, kind=kind, status=status, text=text,
                  token=token, all_records=all_records, limit=limit)  # fmt: skip
    if as_json:
        for row in rows:
            click.echo(json.dumps(row, ensure_ascii=False))
        return
    if not rows:
        click.echo("No experiment matches.")
        return
    for row in rows:
        tokens = ", ".join(repr(t) for t in row["activating_tokens"][:5])
        click.echo(f"{row['model_name']} {unit_label(row)}  {row['unit_status'] or '—'}  "
                   f"{row['experiment_id']}")  # fmt: skip
        click.echo(f"    {(row['explanation'] or '').strip()[:160]}")
        if tokens:
            click.echo(f"    tokens: {tokens}")
