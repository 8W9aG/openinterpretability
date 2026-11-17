"""A model's most recent experiments, for its page (issue #3276).

The home page lists the newest experiments of the whole registry. A model's page lists
its own, from the tree, because model pages render before the registry's statistics.
"""

from __future__ import annotations

from datetime import UTC

RECENT_ON_MODEL = 10  # experiments a model page lists


def unit_label(ident) -> str:
    """``N12``, ``H3``, ``F40`` or ``S0123456789ab``; the template adds the layer and pole."""
    kind = ident.kind
    if kind == "attn_head":
        return f"H{ident.head_idx}"
    if kind == "dictionary_feature":
        return f"F{ident.feature_idx}"
    if kind == "superposition":
        return f"S{ident.superposition_id}"
    return f"N{ident.neuron_idx}"


def recent_experiments(model, n: int = RECENT_ON_MODEL) -> list[dict]:
    """The model's ``n`` newest experiments, whatever their verdict, newest first."""
    found = [e for layer in model.layers for unit in layer.units for e in unit.experiments]
    found.sort(key=lambda e: (e.record.verdict.timestamp, e.id), reverse=True)
    rows = []
    for e in found[:n]:
        verdict, ident = e.record.verdict, e.record.submission.identifier
        when = verdict.timestamp.astimezone(UTC)
        rows.append({
            "href": e.href, "layer": ident.layer, "unit": unit_label(ident),
            "pole": getattr(ident, "pole", None), "status": verdict.status,
            "conclusiveness": e.sig.conclusiveness, "evidence_direction": e.sig.direction,
            "self_reviewed": e.record.self_reviewed, "remeasurement": e.reuses is not None,
            "explanation": e.record.submission.natural_language_explanation,
            "reviewed_iso": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reviewed_date": when.strftime("%Y-%m-%d"),
        })  # fmt: skip
    return rows
