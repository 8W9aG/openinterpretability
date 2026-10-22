"""Bundle every experiment into a columnar registry for bulk analysis.

One row per experiment, whatever its verdict (``is_latest`` marks the newest
experiment for each neuron pole or head). ``kind`` tells the units apart: a neuron
pole row has ``neuron_idx`` and ``pole``, an attention head row ``head_idx``. Writes
two copies of the same table:

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

from openinterp.builder.tree import Experiment
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
        ("experiment_id", pa.string()),
        ("slug", pa.string()),
        ("is_latest", pa.bool_()),
        ("model_name", pa.string()),
        ("kind", pa.string()),
        ("layer", pa.int32()),
        ("neuron_idx", pa.int32()),
        ("pole", pa.string()),
        ("head_idx", pa.int32()),
        ("model_num_layers", pa.int32()),
        ("model_mlp_width", pa.int32()),
        ("model_n_heads", pa.int32()),
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
        ("causal_target", pa.string()),
        ("tests_total", pa.int32()),
        ("tests_passed", pa.int32()),
        ("submitting_agent_model", pa.string()),
        ("reviewer_agent_model", pa.string()),
        ("self_reviewed", pa.bool_()),
        ("reviewed_at", pa.timestamp("us", tz="UTC")),
        ("p_useful", pa.float64()),
        ("conclusiveness", pa.float64()),
        ("evidence_direction", pa.string()),
        ("reuses_probes_of", pa.string()),
        ("supersedes", pa.string()),
        ("effective_hard_negatives", pa.int32()),
        ("effective_holdouts", pa.int32()),
        ("confound_baseline", pa.string()),
        ("confound_baseline_accuracy", pa.float64()),
        ("token_r2", pa.float64()),
        ("position_r2", pa.float64()),
        ("top_partner_neuron", pa.int32()),
        ("top_partner_pole", pa.string()),
        ("top_partner_lift", pa.float64()),
        ("top_partner_spearman", pa.float64()),
        ("redundancy_ratio", pa.float64()),
        ("jacobian_promoted", pa.list_(pa.string())),
        ("logit_promoted", pa.list_(pa.string())),
        ("lens_overlap", pa.float64()),
        ("lens_direction_cosine", pa.float64()),
        ("jacobian_kurtosis", pa.float64()),
        ("current_token_share", pa.float64()),
        ("context_mean_distance", pa.float64()),
        ("context_ablation_agreement", pa.float64()),
        ("detection_score", pa.float64()),
        ("fuzzing_score", pa.float64()),
        ("embedding_auroc", pa.float64()),
        ("previous_token_score", pa.float64()),
        ("duplicate_token_score", pa.float64()),
        ("induction_score", pa.float64()),
        ("record_json", pa.string()),
    ]
)


def record_to_row(record: AuditRecord) -> dict:
    sub, verdict = record.submission, record.verdict
    ident = sub.identifier
    causal = verdict.causal_metrics if verdict else None
    tests = verdict.adversarial_test_results if verdict else []
    is_head = ident.kind == "attn_head"
    return {
        "slug": ident.slug,
        "model_name": ident.model_name,
        "kind": ident.kind,
        "layer": ident.layer,
        "neuron_idx": None if is_head else ident.neuron_idx,
        "pole": ident.pole,
        "head_idx": ident.head_idx if is_head else None,
        "model_num_layers": sub.model_num_layers,
        "model_mlp_width": sub.model_mlp_width,
        "model_n_heads": sub.model_n_heads,
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
        "causal_target": causal.target if causal else None,
        "tests_total": len(tests),
        "tests_passed": sum(t.passed for t in tests),
        "submitting_agent_model": sub.submitting_agent_model,
        "reviewer_agent_model": verdict.reviewer_agent_model if verdict else None,
        "self_reviewed": record.self_reviewed if verdict else None,
        "reviewed_at": verdict.timestamp if verdict else None,
        "record_json": record.model_dump_json(),
    }


def experiment_row(exp: Experiment) -> dict:
    sub, verdict = exp.record.submission, exp.record.verdict
    metrics, confounds, joint = verdict.metrics, sub.confounds, verdict.joint_ablation
    lens, scores, head = sub.vocabulary_lens, verdict.explanation_scores, sub.head_profile
    top = sub.correlated_neurons[0] if sub.correlated_neurons else None
    attribution = sub.input_attribution
    if attribution is not None and not attribution.examples:
        attribution = None  # every example was skipped: nothing measured
    return {
        **record_to_row(exp.record),
        "experiment_id": exp.id,
        "is_latest": exp.is_latest,
        "p_useful": exp.sig.p_useful,
        "conclusiveness": exp.sig.conclusiveness,
        "evidence_direction": exp.sig.direction,
        "reuses_probes_of": exp.reuses,
        "supersedes": exp.record.submission.supersedes,
        "effective_hard_negatives": exp.sig.negatives.clusters,
        "effective_holdouts": exp.sig.holdouts.clusters,
        "confound_baseline": metrics.confound_baseline if metrics else None,
        "confound_baseline_accuracy": metrics.confound_baseline_accuracy if metrics else None,
        "token_r2": confounds.token_r2 if confounds else None,
        "position_r2": confounds.position_r2 if confounds else None,
        "top_partner_neuron": top.neuron_idx if top else None,
        "top_partner_pole": top.pole if top else None,
        "top_partner_lift": top.coactivation_lift_lower if top else None,
        "top_partner_spearman": top.spearman if top else None,
        "redundancy_ratio": joint.redundancy_ratio if joint else None,
        "jacobian_promoted": lens.jacobian_promoted if lens else None,
        "logit_promoted": lens.logit_promoted if lens else None,
        "lens_overlap": lens.overlap if lens else None,
        "lens_direction_cosine": lens.direction_cosine if lens else None,
        "jacobian_kurtosis": lens.jacobian_kurtosis if lens else None,
        "current_token_share": attribution.current_token_share if attribution else None,
        "context_mean_distance": attribution.mean_distance if attribution else None,
        "context_ablation_agreement": attribution.ablation_agreement if attribution else None,
        "detection_score": _balanced(scores.detection if scores else None),
        "fuzzing_score": _balanced(scores.fuzzing if scores else None),
        "embedding_auroc": scores.embedding_auroc if scores else None,
        "previous_token_score": head.previous_token_score if head else None,
        "duplicate_token_score": head.duplicate_token_score if head else None,
        "induction_score": head.induction_score if head else None,
    }


def _balanced(score) -> float | None:
    return score.balanced_accuracy if score else None


def records_to_table(experiments: Sequence[Experiment]) -> pa.Table:
    rows = sorted(
        (experiment_row(e) for e in experiments),
        key=lambda row: (
            row["model_name"],
            row["layer"],
            row["kind"] != "mlp_neuron_pole",
            row["neuron_idx"] if row["head_idx"] is None else row["head_idx"],
            row["pole"] or "",
            row["experiment_id"],
        ),
    )
    return pa.Table.from_pylist(rows, schema=REGISTRY_SCHEMA)


def export_registry(experiments: Sequence[Experiment], output_dir: Path) -> Path:
    """Write the registry parquet, a content-addressed snapshot and a manifest.

    Returns the path of ``data/registry.parquet``.
    """
    data_dir = Path(output_dir) / "data"
    (data_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    latest = data_dir / "registry.parquet"
    pq.write_table(records_to_table(experiments), latest, compression="zstd")

    digest = hashlib.sha256(latest.read_bytes()).hexdigest()[:16]
    snapshot = data_dir / "snapshots" / f"registry-{digest}.parquet"
    snapshot.write_bytes(latest.read_bytes())

    manifest = {
        "latest": "data/registry.parquet",
        "snapshot": snapshot.relative_to(output_dir).as_posix(),
        "sha256_prefix": digest,
        "experiments": len(experiments),
        "updated_at": max(
            (e.record.verdict.timestamp.isoformat() for e in experiments), default=None
        ),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return latest
