"""Recompute a record's verdict from what it stores, without a model.

A record stores each probe's measured result, the policy, the confound baseline, the
active share and whether the review was independent. ``review_metrics`` and ``decide``
need nothing else. So a change to the gates that would change an old record's verdict
shows here, without a model run. ``openinterp check`` calls ``verdict_drift`` for
every record it reads.

Only records judged under the current ``PROTOCOL_VERSION`` must match. Older protocols had
older rules (docs/PROTOCOL.md §6). For example, protocol 0.8 added the
``not_default_on`` gate, and it let a tie with the confound baseline pass. A pull request that bumps ``PROTOCOL_VERSION`` makes today's records older ones,
so a deliberate change to the rule doesn't fail the check.
"""

from __future__ import annotations

from openinterp.agents.reviewer import decide, review_metrics
from openinterp.core.schemas import PROTOCOL_VERSION, AuditRecord


def recompute(record: AuditRecord):
    """Return the metrics and status that the current code gives the stored results.

    Returns ``None`` when the record has no verdict, metrics or policy to recompute from.
    """
    verdict = record.verdict
    if verdict is None or verdict.metrics is None or verdict.policy is None:
        return None
    stored = verdict.metrics
    baseline = None
    if stored.confound_baseline is not None and stored.confound_baseline_accuracy is not None:
        baseline = (stored.confound_baseline, stored.confound_baseline_accuracy)
    metrics = review_metrics(
        verdict.adversarial_test_results,
        verdict.policy,
        independent=stored.independent_review,
        baseline=baseline,
        active_share=stored.active_share,
        first_token_floor=stored.first_token_floor,
    )
    return metrics, decide(metrics, verdict.policy)


def verdict_drift(record: AuditRecord) -> str | None:
    """Describe how the recomputed verdict differs from the stored one, or return ``None``.

    Records from an older protocol, and records without stored metrics, always return
    ``None`` (see the module docstring).
    """
    verdict = record.verdict
    if verdict is None or verdict.policy is None:
        return None
    if verdict.policy.protocol_version != PROTOCOL_VERSION:
        return None
    result = recompute(record)
    if result is None:
        return None
    metrics, status = result
    stored = verdict.metrics
    diffs = []
    if status != verdict.status:
        diffs.append(f"status {verdict.status} -> {status}")
    changed = sorted(
        g
        for g in set(metrics.gates) | set(stored.gates)
        if metrics.gates.get(g) != stored.gates.get(g)
    )
    if changed:
        diffs.append("gates " + ", ".join(changed))
    if metrics.balanced_accuracy != stored.balanced_accuracy:
        diffs.append(f"balanced accuracy {stored.balanced_accuracy} -> {metrics.balanced_accuracy}")
    if not diffs:
        return None
    return (
        "the current code gives this record a different verdict from its stored results ("
        + "; ".join(diffs)
        + "). Fix the regression, or bump PROTOCOL_VERSION if the rule changed on purpose."
    )
