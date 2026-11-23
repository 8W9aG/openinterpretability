"""Bundle every experiment into a columnar registry for bulk analysis.

One row per experiment, whatever its verdict. ``unit_status`` is the standing of the row's
neuron pole, head or feature (``docs/PROTOCOL.md`` §3 step 9). It is CONTESTED when the
unit's reviews disagree, and is repeated on each of its rows. ``counts_toward_status``
marks the rows that the standing rests on. ``errata`` lists the ids of the errata
(``wiki/errata.jsonl``) that cover the row's record. ``is_latest`` marks the newest experiment for
each unit, which is not necessarily one of those.

``kind`` tells the units apart:

- A neuron pole row has ``neuron_idx`` and ``pole``.
- An attention head row has ``head_idx``.
- A dictionary feature row (the learned-basis baseline) has ``dictionary`` and
  ``feature_idx``.
- A superposition row has ``superposition_id`` and ``superposition_members``: the indices
  of its member neurons in the layer, in index order.

Filter on ``kind`` before computing native-neuron statistics. Writes two copies of the
same table:

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
import shutil
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from openinterpretability.builder.draws_export import export_draws  # noqa: F401  (re-exported)
from openinterpretability.builder.slim import full_records
from openinterpretability.builder.tree import SIDES, Experiment, mlp_widths
from openinterpretability.core.confounds import MATCH_LABEL_SETS, NO_MATCH
from openinterpretability.core.schemas import AuditRecord, CorpusSource

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
        ("errata", pa.list_(pa.string())),
        ("model_name", pa.string()),
        ("kind", pa.string()),
        ("layer", pa.int32()),
        ("neuron_idx", pa.int32()),
        ("pole", pa.string()),
        ("head_idx", pa.int32()),
        ("dictionary", pa.string()),
        ("feature_idx", pa.int32()),
        ("superposition_id", pa.string()),
        ("superposition_members", pa.list_(pa.int32())),
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
        ("protocol_version", pa.string()),
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
        ("agent_product", pa.string()),
        ("host", pa.string()),
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
        ("personal_data_r2", pa.float64()),
        ("personal_data_mi_bits", pa.float64()),
        ("personal_data_active_tokens", pa.int64()),
        ("toxicity_r2", pa.float64()),
        ("toxicity_mi_bits", pa.float64()),
        ("toxicity_active_tokens", pa.int64()),
        ("hate_speech_r2", pa.float64()),
        ("hate_speech_mi_bits", pa.float64()),
        ("hate_speech_active_tokens", pa.int64()),
        ("names_r2", pa.float64()),
        ("names_mi_bits", pa.float64()),
        ("names_active_tokens", pa.int64()),
        ("postal_addresses_r2", pa.float64()),
        ("postal_addresses_mi_bits", pa.float64()),
        ("postal_addresses_active_tokens", pa.int64()),
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
        ("ov_copying_score", pa.float64()),
        ("ov_copy_share", pa.float64()),
        ("ov_suppression_share", pa.float64()),
        ("dictionary_explained_variance", pa.float64()),
        ("dictionary_mean_l0", pa.float64()),
        ("dictionary_dead_share", pa.float64()),
        ("feature_density", pa.float64()),
        ("record_json", pa.string()),
    ]
)


def record_to_row(record: AuditRecord, *, with_json: bool = True) -> dict:
    """The registry columns that come from ``record`` alone.

    ``with_json=False`` leaves out ``record_json``, the whole record as JSON (#4652).
    """
    sub = record.submission
    row = {
        **_unit_columns(sub.identifier, sub.superposition_profile),
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
    }
    if with_json:
        row["record_json"] = record.model_dump_json()
    return row


def _unit_columns(ident, superposition) -> dict:
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
        "superposition_id": ident.superposition_id if kind == "superposition" else None,
        "superposition_members": superposition.neurons if superposition else None,
    }


def _verdict_columns(record: AuditRecord) -> dict:
    verdict = record.verdict
    causal = verdict.causal_metrics if verdict else None
    # A one-token prompt (protocol 0.8) is recorded but not measured, so it isn't a test.
    tests = [t for t in verdict.adversarial_test_results if t.measured] if verdict else []
    return {
        "status": verdict.status if verdict else None,
        # The decision rule the verdict was judged under (docs/PROTOCOL.md §6).
        "protocol_version": verdict.policy.protocol_version if verdict and verdict.policy else None,
        "confidence_score": verdict.confidence_score if verdict else None,
        **_columns(causal, CAUSAL_COLUMNS),
        "tests_total": len(tests),
        "tests_passed": sum(t.passed for t in tests),
        "submitting_agent_model": record.submission.submitting_agent_model,
        "reviewer_agent_model": verdict.reviewer_agent_model if verdict else None,
        "submitting_contributor": record.submission.contributor,
        "reviewer_contributor": verdict.contributor if verdict else None,
        **_product(record),
        "self_reviewed": record.self_reviewed if verdict else None,
        "reviewed_at": verdict.timestamp if verdict else None,
    }


def _product(record) -> dict:
    """The agent product and host, from the verdict's environment, else the scan's."""
    verdict, provenance = record.verdict, record.submission.provenance
    found = [verdict and verdict.environment, provenance and provenance.environment]
    env = next((e for e in found if e is not None), None)
    return {"agent_product": env and env.agent_product, "host": env and env.host}


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
OV_COLUMNS = {  # the head's OV lens (#5062)
    "ov_copying_score": "copying_score",
    "ov_copy_share": "copy_share",
    "ov_suppression_share": "suppression_share",
}
DICTIONARY_COLUMNS = {
    "dictionary_explained_variance": "explained_variance",
    "dictionary_mean_l0": "mean_l0",
    "dictionary_dead_share": "dead_share",
    "feature_density": "density",
}


def experiment_row(exp: Experiment, corpus_sources: Sources = None, *, with_json=True) -> dict:
    """Return one experiment's row (``with_json``: see :func:`record_to_row`).

    ``corpus_sources`` is committed sidecars by SHA-256
    (:func:`openinterpretability.corpus.load_sources`). They give the corpus of records that didn't
    record one, marked as inferred.
    """
    sub, verdict = exp.record.submission, exp.record.verdict
    metrics, scores = verdict.metrics, verdict.explanation_scores
    top = sub.correlated_neurons[0] if sub.correlated_neurons else None
    attribution = sub.input_attribution
    if attribution is not None and not attribution.examples:
        attribution = None  # every example was skipped: nothing measured
    return {
        **record_to_row(exp.record, with_json=with_json),
        **_standing_columns(exp),
        **_corpus_columns(sub.provenance, corpus_sources or {}),
        "balanced_accuracy": metrics.balanced_accuracy if metrics else None,
        "effective_hard_negatives": exp.sig.negatives.clusters,
        "effective_holdouts": exp.sig.holdouts.clusters,
        "confound_baseline": metrics.confound_baseline if metrics else None,
        "confound_baseline_accuracy": metrics.confound_baseline_accuracy if metrics else None,
        **_columns(sub.confounds, CONFOUND_COLUMNS),
        **_match_columns(sub.confounds),
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
        **_columns(sub.head_profile and sub.head_profile.ov_lens, OV_COLUMNS),
        **_columns(sub.dictionary_profile, DICTIONARY_COLUMNS),
    }


def _standing_columns(exp: Experiment) -> dict:
    return {
        "experiment_id": exp.id,
        "is_latest": exp.is_latest,
        "unit_status": exp.standing.status if exp.standing else None,
        "counts_toward_status": exp.counts,
        "errata": [erratum.id for erratum in exp.errata],
        "p_useful": exp.sig.p_useful,
        "conclusiveness": exp.sig.conclusiveness,
        "evidence_direction": exp.sig.direction,
        "reuses_probes_of": exp.reuses,
        "supersedes": exp.record.submission.supersedes,
        "sister_of": exp.record.submission.sister_of,
    }


def _match_columns(confounds) -> dict:
    """One label set's measures per match label set, so a query can find such units."""
    out = {}
    for name in MATCH_LABEL_SETS:
        found = ((confounds.labels or {}) if confounds else {}).get(name)
        key = name.replace(" ", "_")
        active = sum(s.active for s in found.levels if s.label != NO_MATCH) if found else None
        out[f"{key}_r2"] = found.r2 if found else None
        out[f"{key}_mi_bits"] = found.mutual_information_bits if found else None
        out[f"{key}_active_tokens"] = active
    return out


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


UNIT_INDEX_COLUMNS = ("neuron_idx", "head_idx", "feature_idx")  # a superposition has none


BATCH_ROWS = 64  # rows per Arrow batch: one batch's full records are in memory at a time


def records_to_table(
    experiments: Sequence[Experiment], corpus_sources: Sources = None, *, with_json: bool = True
) -> pa.Table:
    """The registry table: one row per experiment, in a fixed order (:func:`record_batches`)."""
    schema = registry_schema(with_json)
    return pa.Table.from_batches(list(record_batches(experiments, corpus_sources, with_json)),
                                 schema=schema)  # fmt: skip


def registry_schema(with_json: bool = True) -> pa.Schema:
    """The registry's columns, without ``record_json`` for a table only the statistics read."""
    if with_json:
        return REGISTRY_SCHEMA
    return REGISTRY_SCHEMA.remove(REGISTRY_SCHEMA.get_field_index("record_json"))


def record_batches(experiments: Sequence[Experiment], corpus_sources: Sources = None,
                   with_json: bool = True) -> Iterator[pa.RecordBatch]:  # fmt: skip
    """The registry's rows in their fixed order, :data:`BATCH_ROWS` at a time.

    The ``model_mlp_width`` column is filled for old records that do not give a width.
    ``record_json`` keeps the signed record as it is, or is left out (``with_json=False``).
    Each batch loads its records in full and adds their JSON. So only one batch's JSON is
    in memory at a time. Holding all of it took about 3 GiB in a build (#4652).
    """
    widths = mlp_widths(e.record for e in experiments)
    order = sorted(experiments, key=_experiment_order)  # from the unit, not a whole row
    schema = registry_schema(with_json)
    for start in range(0, len(order), BATCH_ROWS):
        chunk = order[start : start + BATCH_ROWS]
        with full_records(chunk):  # a slim record lacks some columns' evidence (slim.py)
            full = [_with_width(experiment_row(e, corpus_sources, with_json=with_json), widths)
                    for e in chunk]  # fmt: skip
        yield pa.RecordBatch.from_pylist(full, schema=schema)


def _row_order(row: dict) -> tuple:
    """The fixed order of the registry's rows: by unit, then experiment."""
    return (
        row["model_name"],
        row["layer"],
        list(SIDES).index(row["kind"]),
        row["dictionary"] or row["superposition_id"] or "",
        next((row[c] for c in UNIT_INDEX_COLUMNS if row[c] is not None), 0),
        row["pole"] or "",
        row["experiment_id"],
    )


def _experiment_order(e: Experiment) -> tuple:
    """:func:`_row_order` of ``e``'s row, from its unit's columns alone."""
    sub = e.record.submission
    return _row_order(
        _unit_columns(sub.identifier, sub.superposition_profile) | {"experiment_id": e.id}
    )


def _with_width(row: dict, widths: dict[str, int]) -> dict:
    return {**row, "model_mlp_width": row["model_mlp_width"] or widths.get(row["model_name"])}


LATEST = "data/registry.parquet"
"""The registry parquet's path under the site: the build writes it, ``query`` reads it."""


def export_registry(
    experiments: Sequence[Experiment],
    output_dir: Path,
    corpus_sources: Sources = None,
    *,
    with_json: bool = True,
) -> Path:
    """Write the registry parquet, a snapshot and a manifest; return the parquet's path."""
    data_dir = Path(output_dir) / "data"
    (data_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    latest = Path(output_dir) / LATEST
    # One row group per batch, so no column is ever whole in memory (#4652).
    with pq.ParquetWriter(latest, registry_schema(with_json), compression="zstd") as writer:
        for batch in record_batches(experiments, corpus_sources, with_json):
            writer.write_batch(batch)
            pa.default_memory_pool().release_unused()  # Arrow keeps freed buffers otherwise

    with latest.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()[:16]
    snapshot = data_dir / "snapshots" / f"registry-{digest}.parquet"
    shutil.copyfile(latest, snapshot)

    manifest = {
        "latest": LATEST,
        "snapshot": snapshot.relative_to(output_dir).as_posix(),
        "sha256_prefix": digest,
        "experiments": len(experiments),
        "updated_at": max(
            (e.record.verdict.timestamp.isoformat() for e in experiments), default=None
        ),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return latest
