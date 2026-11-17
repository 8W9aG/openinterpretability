"""How far float32 moves a head's per-token strengths (issue #3956).

For each head record, re-measure every top example on this machine in float32 and in
float64. Report, as fractions of the record's scale (its top peak), the largest
|re-measured − recorded| and the largest |float32 − float64|. The head reproduction
tolerance (``reproduce.HEAD_TOLERANCE``) is set from these numbers.

    uv run python scripts/head_tolerance.py \\
        --out docs/analyses/head-reproduce-tolerance.jsonl [RECORD ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from openinterpretability.cli import _load_unit_runner
from openinterpretability.core.paths import DEFAULT_WIKI_DIR, experiments_dir
from openinterpretability.core.schemas import AuditRecord, HeadIdentifier
from openinterpretability.harness.units import unit_of


def head_records(wiki_dir: Path = DEFAULT_WIKI_DIR) -> list[Path]:
    """Every head record in the registry, by path."""
    return sorted(experiments_dir(wiki_dir).glob("*/L*/H*/*.json"))


def measure(path: Path) -> dict:
    """The largest deviations of one record's top examples, as fractions of its scale."""
    record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
    sub = record.submission
    ident = sub.identifier
    if not isinstance(ident, HeadIdentifier):
        raise SystemExit(f"{path} is not a head record")
    revision = sub.provenance.model_revision if sub.provenance else None
    r32 = _load_unit_runner(ident, "cpu", "float32", revision)
    r64 = _load_unit_runner(ident, "cpu", "float32", revision)
    r64.model.double()
    unit = unit_of(ident, head_profile=sub.head_profile)
    scale = max(e.max_activation for e in sub.top_examples)
    recorded = here = rounding = 0.0
    for ex in sub.top_examples:
        a32 = unit.strengths(r32, r32.encode(ex.text))[0].double()
        a64 = unit.strengths(r64, r64.encode(ex.text))[0].double()
        rec = torch.tensor(ex.activations, dtype=torch.float64)
        n = min(len(rec), len(a32))
        here = max(here, (a32[:n] - rec[:n]).abs().max().item())
        recorded = max(recorded, (rec[:n] - a64[:n]).abs().max().item())
        rounding = max(rounding, (a32[:n] - a64[:n]).abs().max().item())
    env = sub.provenance.environment if sub.provenance else None
    return {
        "record": str(path),
        "model": ident.model_name,
        "dtype": sub.provenance.dtype if sub.provenance else None,
        "scanned_on": env.device_name if env else None,
        "scale": round(scale, 6),
        "remeasured_vs_recorded": round(here / scale, 6),
        "recorded_vs_float64": round(recorded / scale, 6),
        "float32_vs_float64": round(rounding / scale, 6),
    }


def main() -> None:
    """Measure the named records, or every head record, and write one row each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [measure(p) for p in (args.records or head_records())]
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for r in rows:
        print(r["record"], r["remeasured_vs_recorded"], r["float32_vs_float64"])


if __name__ == "__main__":
    main()
