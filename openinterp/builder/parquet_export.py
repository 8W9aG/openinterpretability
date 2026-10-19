"""Bundle audit records into a columnar registry for bulk analysis.

Writes two copies of the same table:

* ``data/registry.parquet`` -- stable URL, always the latest build (short cache).
* ``data/snapshots/registry-<sha256>.parquet`` -- content-addressed, cached forever.

``data/manifest.json`` points at the current snapshot.

Output is deterministic: identical records produce byte-identical files, so an
unchanged registry deploys as a no-op.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from openinterp.core.schemas import AuditRecord

COUNTERFACTUAL_TYPE = pa.struct(
    [
        ("base_prompt", pa.string()),
        ("counterfactual_prompt", pa.string()),
        ("expected_delta", pa.float64()),
        ("observed_delta", pa.float64()),
    ]
)

REGISTRY_SCHEMA = pa.schema(
    [
        ("slug", pa.string()),
        ("model_name", pa.string()),
        ("layer", pa.int32()),
        ("neuron_idx", pa.int32()),
        ("pole", pa.string()),
        ("model_num_layers", pa.int32()),
        ("explanation", pa.string()),
        ("activating_tokens", pa.list_(pa.string())),
        ("is_lookahead", pa.bool_()),
        ("target_upcoming_tokens", pa.list_(pa.string())),
        ("kl_divergence", pa.float64()),
        ("max_activation", pa.float64()),
        ("counterfactuals", pa.list_(COUNTERFACTUAL_TYPE)),
        ("status", pa.string()),
        ("confidence_score", pa.float64()),
        ("sufficiency_score", pa.float64()),
        ("necessity_score", pa.float64()),
        ("hydra_mitigated", pa.bool_()),
        ("tests_total", pa.int32()),
        ("tests_passed", pa.int32()),
        ("submitting_agent_model", pa.string()),
        ("reviewer_agent_model", pa.string()),
        ("reviewed_at", pa.timestamp("us", tz="UTC")),
        ("record_json", pa.string()),
    ]
)


def record_to_row(record: AuditRecord) -> dict:
    sub, verdict = record.submission, record.verdict
    ident = sub.identifier
    causal = verdict.causal_metrics if verdict else None
    tests = verdict.adversarial_test_results if verdict else []
    return {
        "slug": ident.slug,
        "model_name": ident.model_name,
        "layer": ident.layer,
        "neuron_idx": ident.neuron_idx,
        "pole": ident.pole,
        "model_num_layers": sub.model_num_layers,
        "explanation": sub.natural_language_explanation,
        "activating_tokens": sub.activating_tokens,
        "is_lookahead": sub.lookahead.is_lookahead,
        "target_upcoming_tokens": sub.lookahead.target_upcoming_tokens,
        "kl_divergence": sub.lookahead.kl_divergence,
        "max_activation": sub.max_activation,
        "counterfactuals": [cf.model_dump() for cf in sub.counterfactuals],
        "status": verdict.status if verdict else None,
        "confidence_score": verdict.confidence_score if verdict else None,
        "sufficiency_score": causal.sufficiency_score if causal else None,
        "necessity_score": causal.necessity_score if causal else None,
        "hydra_mitigated": causal.hydra_mitigated if causal else None,
        "tests_total": len(tests),
        "tests_passed": sum(t.passed for t in tests),
        "submitting_agent_model": sub.submitting_agent_model,
        "reviewer_agent_model": verdict.reviewer_agent_model if verdict else None,
        "reviewed_at": verdict.timestamp if verdict else None,
        "record_json": record.model_dump_json(),
    }


def records_to_table(records: Sequence[AuditRecord]) -> pa.Table:
    rows = sorted(
        (record_to_row(r) for r in records),
        key=lambda row: (row["model_name"], row["layer"], row["neuron_idx"], row["pole"]),
    )
    return pa.Table.from_pylist(rows, schema=REGISTRY_SCHEMA)


def export_registry(records: Sequence[AuditRecord], output_dir: Path) -> Path:
    """Write the registry parquet, a content-addressed snapshot and a manifest.

    Returns the path of ``data/registry.parquet``.
    """
    data_dir = Path(output_dir) / "data"
    (data_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    latest = data_dir / "registry.parquet"
    pq.write_table(records_to_table(records), latest, compression="zstd")

    digest = hashlib.sha256(latest.read_bytes()).hexdigest()[:16]
    snapshot = data_dir / "snapshots" / f"registry-{digest}.parquet"
    snapshot.write_bytes(latest.read_bytes())

    manifest = {
        "latest": "data/registry.parquet",
        "snapshot": snapshot.relative_to(output_dir).as_posix(),
        "sha256_prefix": digest,
        "records": len(records),
        "updated_at": max(
            (r.verdict.timestamp.isoformat() for r in records if r.verdict), default=None
        ),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return latest
