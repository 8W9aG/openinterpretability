"""What a revision's proposer sees of the record that it supersedes (issue #3246).

``open-interpretability audit --supersedes <record>`` revises an earlier hypothesis about the same
unit. Without this, the proposer wrote its new hypothesis from the scan alone and could
repeat the old mistake. The brief gives the old hypothesis, its verdict, the gates it
failed and the probes it got wrong. The proposer may see the measurements: only the
reviewer must not.

With ``wiki_dir``, the brief also covers the attempts before it on the unit: it follows each
record's ``supersedes`` back, up to :data:`EARLIER`. So the proposer of an until-verified
draw's turn k sees turns 1 to k-1 (issue #4485). Any revision also sees its unit's history.
"""

from __future__ import annotations

from pathlib import Path

from openinterpretability.core.fence import fence

MAX_WRONG = 12  # wrong probes listed, so a long probe set doesn't flood the prompt
PROBES = ("hard_negative", "positive_holdout")
EARLIER = 4  # attempts before the superseded one that the brief also covers


def previous_attempt(record, wiki_dir: Path | None = None) -> str:
    """Return the brief on ``record`` for the proposer of the hypothesis that supersedes it.

    With ``wiki_dir``, the attempts that ``record`` itself superseded follow, newest first.
    """
    brief = _attempt(record)
    for i, earlier in enumerate(_chain(record, wiki_dir), 2):
        brief += f"\nAttempt {i} before this one:\n" + _attempt(earlier, intro=False)
    return brief


def _chain(record, wiki_dir: Path | None) -> list:
    """The records that ``record`` superseded in turn, newest first, up to :data:`EARLIER`."""
    from openinterpretability.core.paths import experiments_dir
    from openinterpretability.core.schemas import AuditRecord

    out: list = []
    if wiki_dir is None:
        return out
    folder = experiments_dir(Path(wiki_dir)) / record.submission.identifier.slug
    current = record
    while (stamp := current.submission.supersedes) and len(out) < EARLIER:
        path = folder / f"{stamp}.json"
        if not path.exists():
            break
        current = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
        out.append(current)
    return out


def _attempt(record, intro: bool = True) -> str:
    """The brief on one earlier attempt: its hypothesis, verdict, failed gates and wrong probes."""
    verdict = record.verdict
    gates = verdict.metrics.gates if verdict.metrics else {}
    failed = [gate for gate, ok in gates.items() if not ok]
    wrong = [
        r
        for r in verdict.adversarial_test_results
        if r.kind in PROBES and r.measured and not r.passed
    ]
    lines = [
        *(
            [
                "This hypothesis revises an earlier one about this unit. Fix what failed, and "
                "keep what the earlier review confirmed."
            ]
            if intro
            else []
        ),
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
