"""``openinterp wiring-backfill``: wiring for archived records that predate it.

See :mod:`openinterp.wiring_backfill`.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import _load_runner, cli, wiki_dir_option
from openinterp.wiring_backfill import DEFAULT_PATH


@cli.command("wiring-backfill")
@wiki_dir_option()
@click.option(
    "--out",
    default=DEFAULT_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="JSONL of wiring rows; rows already in it are skipped, so a run can resume.",
)
@click.option("--limit", type=int, help="Wire at most this many records.")
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="cpu, cuda, mps or auto. Each record's own model revision and dtype are used.",
)
def wiring_backfill(wiki_dir, out, limit, device):
    """Record the wiring of every archived experiment that has none, in --out.

    Records are write-once and often signed, so they are left alone; `openinterp build`
    shows a row's wiring for its record, marked as backfilled.
    """
    from openinterp.builder.wiki import load_archive
    from openinterp.harness.units import attn_implementation
    from openinterp.wiring_backfill import backfill, sort_rows

    def load(record):
        sub = record.submission
        prov = sub.provenance
        dtype = (prov.dtype if prov else None) or "float32"
        revision = prov.model_revision if prov else None
        return _load_runner(
            sub.identifier.model_name, device, dtype, revision, attn_implementation(sub.identifier)
        )

    wrote = backfill(load_archive(wiki_dir), load, out, limit=limit, echo=click.echo)
    if out.exists():
        sort_rows(out)
    click.echo(f"Wired {wrote} experiments → {out}")
