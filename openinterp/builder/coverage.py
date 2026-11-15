"""Registry coverage for the site: each model's newest `open-interpretability coverage` run.

``openinterp/coverage.py`` measures coverage, which needs the model. This module only
reads the rows that the command wrote, so the site build needs no model and no torch.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_ROWS = Path("docs/analyses/registry-coverage.jsonl")


def load_rows(path: Path = DEFAULT_ROWS) -> dict[str, list[dict]]:
    """Each model's newest run in ``path``, as ``{model: [rows by layer]}``. Empty if missing."""
    path = Path(path)
    if not path.exists():
        return {}
    newest: dict[str, str] = {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        newest[row["model"]] = max(newest.get(row["model"], ""), row["date"])
    out: dict[str, list[dict]] = {}
    for row in rows:
        if row["date"] == newest[row["model"]]:
            out.setdefault(row["model"], []).append(row)
    return {m: sorted(rs, key=lambda r: r["layer"]) for m, rs in out.items()}


def model_summary(rows: list[dict]) -> dict:
    """Loss-weighted shares over a model's layers, for the VERIFIED poles and for random ones."""
    rise = sum(r["loss_ablated"] - r["loss_clean"] for r in rows)
    kept = sum(r["loss_ablated"] - r.get("loss_kept_verified", r["loss_ablated"]) for r in rows)
    rand = sum(
        (r.get("recovered_random") or 0.0) * (r["loss_ablated"] - r["loss_clean"]) for r in rows
    )
    return {
        "model": rows[0]["model"],
        "date": rows[0]["date"],
        "layers": len(rows),
        "verified_poles": sum(r["verified_poles"] for r in rows),
        "poles": sum(2 * r["width"] for r in rows),
        "recovered": kept / rise if rise > 0 else None,
        "recovered_random": rand / rise if rise > 0 else None,
    }
