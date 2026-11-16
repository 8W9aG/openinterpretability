"""``open-interpretability circuits``: circuit ids that survive as the registry grows (#4284)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click

from openinterp.cli import cli
from openinterp.core.paths import DEFAULT_WIKI_DIR


@cli.group(name="circuits")
def circuits_group() -> None:
    """Circuits: clusters of the wiring graph (docs/circuits-and-hubs.md)."""


@circuits_group.command(name="identify")
@click.option(
    "--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default=DEFAULT_WIKI_DIR
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/circuits"),
    show_default=True,
)
@click.option("--model", "only", default="", help="One model slug; default: every model.")
def identify_command(wiki_dir: Path, out_dir: Path, only: str) -> None:
    """Match each model's wiring clusters to its stored circuit ids, and record the changes.

    It appends what changed to data/circuits/<model>.jsonl (openinterp/circuit_ids.py).
    """
    from openinterp import circuit_ids
    from openinterp.builder.tree import build_tree
    from openinterp.builder.wiki import load_archive
    from openinterp.core.errata import load_errata
    from openinterp.wiring_backfill import load_rows

    today = datetime.now(UTC).date().isoformat()
    rows = load_rows()
    for model in build_tree(load_archive(wiki_dir), load_errata(wiki_dir)):
        if only and model.slug != only:
            continue
        path = circuit_ids.path_for(model.slug, out_dir)
        old, ever = circuit_ids.load(path)
        result = circuit_ids.update(old, circuit_ids.clusters(model, rows), today, ever)
        circuit_ids.append(path, result.events)
        kinds = ", ".join(sorted({e["event"] for e in result.events})) or "no change"
        click.echo(f"{model.slug}: {len(result.circuits)} circuits ({kinds})")
