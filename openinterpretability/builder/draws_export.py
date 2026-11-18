"""Export the random draws that no experiment could audit, as ``data/draws.parquet``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DRAW_COLUMNS = {
    "kind": pa.string(),
    "model_name": pa.string(),
    "layer": pa.int32(),
    "index": pa.int32(),
    "pole": pa.string(),
    "dictionary": pa.string(),
    "outcome": pa.string(),
    "active_share": pa.float64(),
    "corpus_sha256": pa.string(),
    "window_tokens": pa.int32(),
    "seed": pa.int64(),
    "protocol_version": pa.string(),
    "drawn_at": pa.timestamp("us", tz="UTC"),
}


def export_draws(draws: Sequence, output_dir: Path) -> Path:
    """Write ``data/draws.parquet``: the random draws no experiment could audit."""
    rows = []
    for d in draws:
        ident, prov = d.identifier, d.provenance
        rows.append(
            {
                "kind": ident.kind,
                "model_name": ident.model_name,
                "layer": ident.layer,
                "index": ident.index,
                "pole": getattr(ident, "pole", None),
                "dictionary": getattr(ident, "dictionary", None),
                "outcome": d.outcome,
                "active_share": d.active_share,
                "corpus_sha256": prov.corpus_sha256,
                "window_tokens": prov.snippet_tokens,
                "seed": prov.seed,
                "protocol_version": d.protocol_version,
                "drawn_at": d.timestamp,
            }
        )
    path = Path(output_dir) / "data" / "draws.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(list(DRAW_COLUMNS.items()))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
    return path
