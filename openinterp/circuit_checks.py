"""Checks of circuit records in ``wiki/circuits/`` (issue #4284), run by ``check_registry``.

Each record must validate as a :class:`~openinterp.core.circuit_records.CircuitRecord`.
It must sit at the path that its model, circuit id and timestamp give. Under the current
circuit rule version, its status and gates must be what :func:`decide` gives from its
stored measures. Write-once applies to it as to every archived record.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from openinterp.core.circuit_records import (
    CIRCUIT_PROTOCOL_VERSION,
    CIRCUITS_WIKI,
    CircuitRecord,
    decide,
    record_path,
)


def check_circuits(wiki_dir: Path, report, scope: set[Path] | None = None) -> None:
    """Add a problem to ``report`` for each circuit record that fails a check."""
    root = wiki_dir / CIRCUITS_WIKI
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json")):
        if scope is not None and path not in scope:
            continue
        try:
            record = CircuitRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            report.add(path, f"not a valid circuit record: {error.errors()[0]['msg']}")
            continue
        if record_path(wiki_dir, record) != path:
            report.add(path, f"should be at {record_path(wiki_dir, record)}")
        verdict = record.verdict
        if verdict.policy.protocol_version == CIRCUIT_PROTOCOL_VERSION:
            status, gates = decide(verdict.measures, verdict.policy)
            if (status, gates) != (verdict.status, verdict.gates):
                report.add(path, f"the measures give {status}, not {verdict.status}")
