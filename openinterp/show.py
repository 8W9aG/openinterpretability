"""Compact summaries of records, submissions, scan evidence and a unit's history.

``openinterp show`` prints these so agents never open a 300 KB record, a 500 KB
``ev.json`` or a 25 KB experiment page just to learn a verdict, the failed gates and
probes, or what a scan found. The files stay the source of truth; this only reads them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from openinterp.core.schemas import (
    ACTIVATION_THRESHOLD,
    AuditRecord,
    NeuronAuditSubmission,
    ScanEvidence,
)

PROMPT_CHARS = 160  # probes and examples are cut to this unless full=True
TOP_EXAMPLES = 5
CONTEXT_TOKENS = 8  # tokens shown before an example's peak


def show(path: Path, full: bool = False) -> str:
    """Summarise ``path``: a record, submission or evidence JSON, or a unit's directory
    (every experiment on it, oldest first, with each page's Discussion)."""
    if path.is_dir():
        return show_unit(path, full)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)  # to tell the kinds apart; strict models validate the text
    if "submission" in data:
        return show_record(AuditRecord.model_validate_json(text), path, full)
    if "natural_language_explanation" in data:
        return show_submission(NeuronAuditSubmission.model_validate_json(text), full)
    if "top_examples" in data:
        return show_evidence(ScanEvidence.model_validate_json(text), full)
    raise ValueError(f"{path} is not a record, submission or scan evidence")


def show_unit(unit_dir: Path, full: bool = False) -> str:
    records = sorted(
        (
            (AuditRecord.model_validate_json(p.read_text(encoding="utf-8")), p)
            for p in unit_dir.glob("*.json")
        ),
        key=lambda rp: rp[0].verdict.timestamp if rp[0].verdict else 0,
    )
    if not records:
        raise ValueError(f"no experiment records in {unit_dir}")
    parts = [f"{len(records)} experiment(s) in {unit_dir}, oldest first"]
    parts += [show_record(record, path, full) for record, path in records]
    return "\n\n".join(parts)


def show_record(record: AuditRecord, path: Path | None = None, full: bool = False) -> str:
    sub, verdict = record.submission, record.verdict
    name = f"{sub.identifier.slug}" + (f" · {path.stem}" if path else "")
    if verdict is None:
        return f"{name} · not reviewed\n{show_submission(sub, full)}"
    status = f"{verdict.status} {verdict.confidence_score:.2f}"
    protocol = verdict.policy.protocol_version if verdict.policy else "< 0.2"
    lines = [
        f"== {name} · {status} (protocol {protocol})"
        + (" · self-reviewed" if record.self_reviewed else "")
    ]
    lines.append(
        f"Proposer {sub.submitting_agent_model} · reviewer {verdict.reviewer_agent_model}"
        + _selection(sub)
        + (f" · supersedes {sub.supersedes}" if sub.supersedes else "")
    )
    lines += _hypothesis(sub)
    lines += _metrics(verdict)
    lines += _failed_probes(record, full)
    if verdict.notes:
        lines.append(f"Notes: {_clip(verdict.notes, full, 400)}")
    if path is not None and (discussion := _discussion(path.with_suffix(".md"))):
        lines.append(f"Discussion: {_clip(discussion, full, 1200)}")
    return "\n".join(lines)


def _metrics(verdict) -> list[str]:
    """Failed gates, the review's rates, and the scores recorded beside them."""
    lines, m = [], verdict.metrics
    if m is not None:
        failed = [g for g, ok in m.gates.items() if not ok]
        lines.append("Failed gates: " + (", ".join(failed) if failed else "none"))
        rates = [
            ("specificity", m.specificity),
            ("sensitivity", m.sensitivity),
            ("balanced accuracy", m.balanced_accuracy),
            ("counterfactuals", m.counterfactual_reproduction),
        ]
        line = " · ".join(f"{k} {v:.2f}" for k, v in rates if v is not None)
        if m.confound_baseline_accuracy is not None:
            line += f" · best baseline ({m.confound_baseline}) {m.confound_baseline_accuracy:.2f}"
        lines.append(line)
    if (s := verdict.explanation_scores) is not None:
        scores = [("detection", s.detection), ("fuzzing", s.fuzzing)]
        lines.append(
            "Explanation scores: "
            + " · ".join(f"{k} {v.balanced_accuracy:.2f}" for k, v in scores if v is not None)
        )
    if (c := verdict.causal_metrics) is not None:
        lines.append(
            f"Causal: sufficiency {c.sufficiency_score:.2f} · necessity {c.necessity_score:.2f}"
        )
    return lines


def show_submission(sub: NeuronAuditSubmission, full: bool = False) -> str:
    lines = [
        f"== {sub.identifier.slug} · submission by {sub.submitting_agent_model}{_selection(sub)}"
    ]
    lines += _hypothesis(sub)
    lines.append(f"Counterfactuals: {len(sub.counterfactuals)}")
    return "\n".join(lines)


def show_evidence(ev: ScanEvidence, full: bool = False) -> str:
    lines = [f"== {ev.identifier.slug} · scan evidence"]
    if (p := ev.provenance) is not None:
        window = f", {p.snippet_tokens}-token windows" if p.snippet_tokens else ""
        lines.append(
            f"Corpus {p.corpus}: {p.n_snippets} snippets, {p.tokens_scanned:,} tokens{window}"
            + (f" · selection {p.selection} (seed {p.seed})" if p.seed is not None else "")
        )
    top = ev.top_examples[0].max_activation if ev.top_examples else 0.0
    line = f"Max activation {top:.3f}"
    if ev.activation_histogram is not None and top > 0:
        share = ev.activation_histogram.share_at_least(ACTIVATION_THRESHOLD * top)
        line += f" · ≥ {ACTIVATION_THRESHOLD:g}×max on at least {share:.1%} of tokens"
    if ev.first_token_floor:
        line += " · first-token floor"
    lines.append(line)
    shown = ev.top_examples if full else ev.top_examples[:TOP_EXAMPLES]
    lines.append(
        f"Top examples ({len(shown)} of {len(ev.top_examples)}), peak token in [brackets]:"
    )
    lines += [f"  {ex.max_activation:7.3f}  {_peak(ex, full)}" for ex in shown]
    if (lens := ev.vocabulary_lens) is not None:
        lines.append(f"Logit lens promotes: {_tokens(lens.logit_promoted)}")
        lines.append(f"Logit lens suppresses: {_tokens(lens.logit_suppressed)}")
    if ev.correlated_neurons:
        sign = {"positive": "+", "negative": "−"}
        lines.append(
            "Co-activating poles: "
            + ", ".join(
                f"N{n.neuron_idx}{sign[n.pole]} (lift ≥ {n.coactivation_lift_lower:.1f})"
                for n in ev.correlated_neurons[:5]
            )
        )
    if (op := ev.opposite_pole) is not None and op.top_examples:
        ex = op.top_examples[0]
        lines.append(f"Opposite pole ({op.pole}) peak {ex.max_activation:.3f}: {_peak(ex, full)}")
    return "\n".join(lines)


def _selection(sub) -> str:
    p = sub.provenance
    if p is None:
        return ""
    return f" · selection {p.selection}" + (f" (seed {p.seed})" if p.seed is not None else "")


def _hypothesis(sub) -> list[str]:
    return [
        f"Hypothesis: {sub.natural_language_explanation}",
        f"Tokens: {_tokens(sub.activating_tokens)} · mechanism "
        + ("lookahead" if sub.lookahead.is_lookahead else "retrospective")
        + (f" · max activation {sub.max_activation:.3f}" if sub.max_activation else ""),
    ]


def _failed_probes(record: AuditRecord, full: bool) -> list[str]:
    verdict = record.verdict
    results = verdict.adversarial_test_results
    failed = [r for r in results if r.measured and not r.passed]
    if not results:
        return []
    line = f"Failed probes: {len(failed)} of {len(results)}"
    peak = record.submission.max_activation
    if verdict.policy is not None and peak:
        line += f" (active line {verdict.policy.activation_threshold * peak:.3f})"
    lines = [line + (":" if failed else "")]
    for r in failed:
        if r.kind == "counterfactual":
            got = f"Δ {_num(r.observed_delta)} vs expected {_num(r.expected_delta)}"
        else:
            expected = "active" if r.expected_active else "inactive"
            got = f"{_num(r.activation)}, expected {expected}"
        lines.append(f"  ✗ {r.kind.replace('_', ' ')}: {got}: {_clip(r.prompt, full)}")
    return lines


def _discussion(page: Path) -> str:
    """The page's Discussion section without the template comment, or ''."""
    if not page.exists():
        return ""
    text = page.read_text(encoding="utf-8").split("\n## Discussion", 1)
    if len(text) < 2:
        return ""
    body = re.sub(r"<!--.*?-->", "", text[1], flags=re.S)
    return " ".join(body.split())


def _peak(ex, full: bool) -> str:
    i = ex.max_token_index
    start = 0 if full else max(0, i - CONTEXT_TOKENS)
    end = len(ex.tokens) if full else i + 3
    before, peak, after = ex.tokens[start:i], ex.tokens[i : i + 1], ex.tokens[i + 1 : end]
    text = "".join(before) + "[" + "".join(peak) + "]" + "".join(after)
    return ("…" if start else "") + text.replace("\n", "⏎")


def _tokens(tokens: list[str], n: int = 10) -> str:
    shown = ", ".join(repr(t) for t in tokens[:n])
    return shown + (f" (+{len(tokens) - n})" if len(tokens) > n else "")


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _clip(text: str, full: bool, chars: int = PROMPT_CHARS) -> str:
    text = " ".join(text.split())
    return text if full or len(text) <= chars else text[: chars - 1] + "…"
