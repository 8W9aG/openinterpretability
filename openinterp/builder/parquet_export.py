"""Bundle every experiment into a columnar registry for bulk analysis.

One row per experiment, whatever its verdict. ``unit_status`` is the standing of the row's
neuron pole, head or feature (``docs/PROTOCOL.md`` §3 step 9; CONTESTED when its reviews
disagree), repeated on each of its rows, and ``counts_toward_status`` marks the rows it
rests on. ``is_latest`` marks the newest experiment for each, which is not necessarily one
of those. ``kind`` tells the units apart: a
neuron pole row has ``neuron_idx`` and ``pole``, an attention head row ``head_idx``, and a
dictionary feature row (the learned-basis baseline) ``dictionary`` and ``feature_idx``.
Filter on ``kind`` before computing native-neuron statistics. Writes
two copies of the same table:

* ``data/registry.parquet`` -- stable URL, always the latest build (short cache).
* ``data/snapshots/registry-<sha256>.parquet`` -- content-addressed, cached forever.

``data/manifest.json`` points at the current snapshot. ``data/draws.parquet`` has one
row per random draw that could not be audited (dead or default-on, ``wiki/draws/``).

Output is deterministic: identical records produce byte-identical files, so an
unchanged registry deploys as a no-op.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from openinterp.builder.tree import SIDES, Experiment
from openinterp.core.schemas import AuditRecord, CorpusSource

Sources = Mapping[str, CorpusSource] | None

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
        ("unit_status", pa.string()),
        ("counts_toward_status", pa.bool_()),
        ("model_name", pa.string()),
        ("kind", pa.string()),
        ("layer", pa.int32()),
        ("neuron_idx", pa.int32()),
        ("pole", pa.string()),
        ("head_idx", pa.int32()),
        ("dictionary", pa.string()),
        ("feature_idx", pa.int32()),
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
        ("submitting_contributor", pa.string()),
        ("reviewer_contributor", pa.string()),
        ("self_reviewed", pa.bool_()),
        ("reviewed_at", pa.timestamp("us", tz="UTC")),
        ("p_useful", pa.float64()),
        ("conclusiveness", pa.float64()),
        ("evidence_direction", pa.string()),
        ("reuses_probes_of", pa.string()),
        ("supersedes", pa.string()),
        ("sister_of", pa.string()),
        ("selection", pa.string()),
        ("corpus_sha256", pa.string()),
        ("corpus_dataset", pa.string()),
        ("corpus_revision", pa.string()),
        ("corpus_license", pa.string()),
        ("corpus_source_inferred", pa.bool_()),
        ("window_tokens", pa.int32()),
        ("balanced_accuracy", pa.float64()),
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
        ("jacobian_promoted_z", pa.list_(pa.float64())),
        ("logit_promoted_z", pa.list_(pa.float64())),
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
        ("dictionary_explained_variance", pa.float64()),
        ("dictionary_mean_l0", pa.float64()),
        ("dictionary_dead_share", pa.float64()),
        ("feature_density", pa.float64()),
        ("record_json", pa.string()),
    ]
)


def record_to_row(record: AuditRecord) -> dict:
    sub = record.submission
    return {
        **_unit_columns(sub.identifier),
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
        **_verdict_columns(record),
        "record_json": record.model_dump_json(),
    }


def _unit_columns(ident) -> dict:
    kind = ident.kind
    return {
        "slug": ident.slug,
        "model_name": ident.model_name,
        "kind": ident.kind,
        "layer": ident.layer,
        "neuron_idx": ident.neuron_idx if kind == "mlp_neuron_pole" else None,
        "pole": ident.pole,
        "head_idx": ident.head_idx if kind == "attn_head" else None,
        "dictionary": ident.dictionary if kind == "dictionary_feature" else None,
        "feature_idx": ident.feature_idx if kind == "dictionary_feature" else None,
    }


def _verdict_columns(record: AuditRecord) -> dict:
    verdict = record.verdict
    causal = verdict.causal_metrics if verdict else None
    # A one-token prompt (protocol 0.8) is recorded but not measured, so it isn't a test.
    tests = [t for t in verdict.adversarial_test_results if t.measured] if verdict else []
    return {
        "status": verdict.status if verdict else None,
        "confidence_score": verdict.confidence_score if verdict else None,
        **_columns(causal, CAUSAL_COLUMNS),
        "tests_total": len(tests),
        "tests_passed": sum(t.passed for t in tests),
        "submitting_agent_model": record.submission.submitting_agent_model,
        "reviewer_agent_model": verdict.reviewer_agent_model if verdict else None,
        "submitting_contributor": record.submission.contributor,
        "reviewer_contributor": verdict.contributor if verdict else None,
        "self_reviewed": record.self_reviewed if verdict else None,
        "reviewed_at": verdict.timestamp if verdict else None,
    }


def _columns(obj, columns: dict[str, str]) -> dict:
    """``column -> obj.<attribute>`` for each pair in ``columns``; all ``None`` without ``obj``."""
    return {col: getattr(obj, attr) if obj else None for col, attr in columns.items()}


# Parquet column -> attribute, per object a row reads; the order is the column order.
CAUSAL_COLUMNS = {
    "sufficiency_score": "sufficiency_score",
    "necessity_score": "necessity_score",
    "hydra_mitigated": "hydra_mitigated",
    "causal_target": "target",
}
CONFOUND_COLUMNS = {"token_r2": "token_r2", "position_r2": "position_r2"}
PARTNER_COLUMNS = {
    "top_partner_neuron": "neuron_idx",
    "top_partner_pole": "pole",
    "top_partner_lift": "coactivation_lift_lower",
    "top_partner_spearman": "spearman",
}
LENS_COLUMNS = {
    "jacobian_promoted": "jacobian_promoted",
    "logit_promoted": "logit_promoted",
    "jacobian_promoted_z": "jacobian_promoted_z",
    "logit_promoted_z": "logit_promoted_z",
    "lens_overlap": "overlap",
    "lens_direction_cosine": "direction_cosine",
    "jacobian_kurtosis": "jacobian_kurtosis",
}
ATTRIBUTION_COLUMNS = {
    "current_token_share": "current_token_share",
    "context_mean_distance": "mean_distance",
    "context_ablation_agreement": "ablation_agreement",
}
HEAD_COLUMNS = {
    "previous_token_score": "previous_token_score",
    "duplicate_token_score": "duplicate_token_score",
    "induction_score": "induction_score",
}
DICTIONARY_COLUMNS = {
    "dictionary_explained_variance": "explained_variance",
    "dictionary_mean_l0": "mean_l0",
    "dictionary_dead_share": "dead_share",
    "feature_density": "density",
}


def experiment_row(exp: Experiment, corpus_sources: Sources = None) -> dict:
    """``corpus_sources``: committed sidecars by SHA-256 (:func:`openinterp.corpus.load_sources`),
    which give the corpus of records that didn't record one, marked as inferred."""
    sub, verdict = exp.record.submission, exp.record.verdict
    metrics, scores = verdict.metrics, verdict.explanation_scores
    top = sub.correlated_neurons[0] if sub.correlated_neurons else None
    attribution = sub.input_attribution
    if attribution is not None and not attribution.examples:
        attribution = None  # every example was skipped: nothing measured
    return {
        **record_to_row(exp.record),
        **_standing_columns(exp),
        **_corpus_columns(sub.provenance, corpus_sources or {}),
        "balanced_accuracy": metrics.balanced_accuracy if metrics else None,
        "effective_hard_negatives": exp.sig.negatives.clusters,
        "effective_holdouts": exp.sig.holdouts.clusters,
        "confound_baseline": metrics.confound_baseline if metrics else None,
        "confound_baseline_accuracy": metrics.confound_baseline_accuracy if metrics else None,
        **_columns(sub.confounds, CONFOUND_COLUMNS),
        **_columns(top, PARTNER_COLUMNS),
        "redundancy_ratio": verdict.joint_ablation.redundancy_ratio
        if verdict.joint_ablation
        else None,
        **_columns(sub.vocabulary_lens, LENS_COLUMNS),
        **_columns(attribution, ATTRIBUTION_COLUMNS),
        "detection_score": _balanced(scores.detection if scores else None),
        "fuzzing_score": _balanced(scores.fuzzing if scores else None),
        "embedding_auroc": scores.embedding_auroc if scores else None,
        **_columns(sub.head_profile, HEAD_COLUMNS),
        **_columns(sub.dictionary_profile, DICTIONARY_COLUMNS),
    }


def _standing_columns(exp: Experiment) -> dict:
    return {
        "experiment_id": exp.id,
        "is_latest": exp.is_latest,
        "unit_status": exp.standing.status if exp.standing else None,
        "counts_toward_status": exp.counts,
        "p_useful": exp.sig.p_useful,
        "conclusiveness": exp.sig.conclusiveness,
        "evidence_direction": exp.sig.direction,
        "reuses_probes_of": exp.reuses,
        "supersedes": exp.record.submission.supersedes,
        "sister_of": exp.record.submission.sister_of,
    }


def _corpus_columns(prov, corpus_sources: dict) -> dict:
    """The scan's corpus, inferred from the committed sidecars when the record has none."""
    source = prov.corpus_source if prov else None
    inferred = source is None and prov is not None and prov.corpus_sha256 in corpus_sources
    source = corpus_sources[prov.corpus_sha256] if inferred else source
    return {
        "selection": prov.selection if prov else None,
        "corpus_sha256": prov.corpus_sha256 if prov else None,
        "corpus_dataset": source.dataset if source else None,
        "corpus_revision": source.revision if source else None,
        "corpus_license": source.license if source else None,
        "corpus_source_inferred": inferred if source else None,
        "window_tokens": prov.snippet_tokens if prov else None,
    }


def _balanced(score) -> float | None:
    return score.balanced_accuracy if score else None


def records_to_table(experiments: Sequence[Experiment], corpus_sources: Sources = None) -> pa.Table:
    rows = sorted(
        (experiment_row(e, corpus_sources) for e in experiments),
        key=lambda row: (
            row["model_name"],
            row["layer"],
            list(SIDES).index(row["kind"]),
            row["dictionary"] or "",
            next(row[c] for c in ("neuron_idx", "head_idx", "feature_idx") if row[c] is not None),
            row["pole"] or "",
            row["experiment_id"],
        ),
    )
    return pa.Table.from_pylist(rows, schema=REGISTRY_SCHEMA)


def export_registry(
    experiments: Sequence[Experiment], output_dir: Path, corpus_sources: Sources = None
) -> Path:
    """Write the registry parquet, a content-addressed snapshot and a manifest.

    Returns the path of ``data/registry.parquet``.
    """
    data_dir = Path(output_dir) / "data"
    (data_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    latest = data_dir / "registry.parquet"
    pq.write_table(records_to_table(experiments, corpus_sources), latest, compression="zstd")

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
