"""What a revision's proposer sees of the record that it supersedes (issue #3246).

``openinterp audit --supersedes <record>`` revises an earlier hypothesis about the same
unit. Without this, the proposer wrote its new hypothesis from the scan alone and could
repeat the old mistake. The brief gives the old hypothesis, its verdict, the gates it
failed and the probes it got wrong. The proposer may see the measurements: only the
reviewer must not.
"""

from __future__ import annotations

from openinterp.core.fence import fence

MAX_WRONG = 12  # wrong probes listed, so a long probe set doesn't flood the prompt
PROBES = ("hard_negative", "positive_holdout")


def previous_attempt(record) -> str:
    """Return the brief on ``record`` for the proposer of the hypothesis that supersedes it."""
    verdict = record.verdict
    gates = verdict.metrics.gates if verdict.metrics else {}
    failed = [gate for gate, ok in gates.items() if not ok]
    wrong = [
        r
        for r in verdict.adversarial_test_results
        if r.kind in PROBES and r.measured and not r.passed
    ]
    lines = [
        "This hypothesis revises an earlier one about this unit. Fix what failed, and "
        "keep what the earlier review confirmed.",
        f"Earlier hypothesis:\n{fence(record.submission.natural_language_explanation)}",
        f"Its verdict: {verdict.status}. Failed gates: {', '.join(failed) or 'none'}.",
    ]
    if wrong:
        lines.append(f"Probes it got wrong ({min(len(wrong), MAX_WRONG)} of {len(wrong)}):")
        lines += [_probe_line(r) for r in wrong[:MAX_WRONG]]
    return "\n".join(lines) + "\n"


def _probe_line(result) -> str:
    """One wrong probe: what the hypothesis predicted, and what was measured."""
    predicted = "active" if result.expected_active else "inactive"
    measured = "n/a" if result.activation is None else f"{result.activation:.2f}"
    kind = result.kind.replace("_", " ")
    return f"- {kind}, predicted {predicted}, activation {measured}: {result.prompt!r}"
