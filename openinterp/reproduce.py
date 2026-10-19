"""Reproduce an experiment record from its JSON before it merges (`openinterp reproduce`).

`openinterp check` only validates a record's shape. This re-runs, against the model
revision and dtype the record pins, every measurement its verdict rests on:

1. the scan evidence: each top example's per-token pole strengths, and the scale
   (``max_activation``) that every "active" threshold is a fraction of;
2. each counterfactual pair's observed Δ;
3. the review itself, with the record's stored probes and its own ``ReviewPolicy``
   (no LLM calls): every probe's activation and pass/fail, every gate and the status.

Numbers must agree within ``atol × scale + rtol × |recorded|``. Anything the decision
rule reads (pass/fail, gates, status) must agree exactly. Causal metrics and joint
ablation are reported but not gated, so a difference there is only a warning.

A record judged under another protocol version can't be reproduced by this harness:
its decision rule was different. Re-evaluate it instead (`openinterp reevaluate`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from openinterp.core.schemas import PROTOCOL_VERSION, AuditRecord, RunEnvironment
from openinterp.harness.hooks import NeuronHookRunner, run_environment


@dataclass(frozen=True)
class Tolerance:
    """Allowed |reproduced − recorded| = atol × scale + rtol × |recorded|.

    The defaults absorb float32 differences between devices and library versions
    (records round activations to 5–6 decimals) but not an edited number.
    """

    atol: float = 1e-3  # fraction of the neuron's max activation
    rtol: float = 1e-2

    def close(self, recorded: float | None, reproduced: float | None, scale: float) -> bool:
        if recorded is None or reproduced is None:
            return recorded is None and reproduced is None
        return abs(reproduced - recorded) <= self.atol * scale + self.rtol * abs(recorded)


DEFAULT_TOLERANCE = Tolerance()


@dataclass
class ReproductionReport:
    path: Path | None
    record: AuditRecord
    mismatches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compared: int = 0

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def fail(self, message: str) -> None:
        self.mismatches.append(message)


def pinned_environment(record: AuditRecord) -> tuple[str | None, str]:
    """(model revision, dtype) to load the model with, from the record's own pins."""
    env = record.verdict.environment if record.verdict else None
    prov = record.submission.provenance
    revision = (env and env.model_revision) or (prov and prov.model_revision)
    dtype = (env and env.dtype) or (prov and prov.dtype) or "float32"
    return revision, dtype


def check_protocol(record: AuditRecord) -> str | None:
    """Why this harness can't reproduce ``record``'s verdict, or ``None`` if it can."""
    if record.verdict is None:
        return "record has no verdict"
    policy = record.verdict.policy
    version = policy.protocol_version if policy else "< 0.2"
    if version != PROTOCOL_VERSION:
        return (
            f"judged under protocol {version}, but this harness implements "
            f"{PROTOCOL_VERSION}; reproduce it with that openinterp version, or re-evaluate it"
        )
    return None


def reproduce_record(
    record: AuditRecord,
    runner: NeuronHookRunner,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
    path: Path | None = None,
) -> ReproductionReport:
    from openinterp.agents.proposer import observed_delta
    from openinterp.agents.reviewer import evaluate_submission, probes_from_verdict

    report = ReproductionReport(path=path, record=record)
    reason = check_protocol(record)
    if reason:
        report.fail(reason)
        return report
    sub, verdict = record.submission, record.verdict
    ident = sub.identifier
    _compare_environment(record, run_environment(runner), report)

    scale = sub.max_activation or 0.0
    peak = max((e.max_activation for e in sub.top_examples), default=None)
    if peak is not None and not math.isclose(scale, peak, rel_tol=1e-9, abs_tol=1e-12):
        report.fail(f"max_activation {scale} is not the top example's peak {peak}")

    # 1. Scan evidence.
    for n, ex in enumerate(sub.top_examples, 1):
        strengths = runner.neuron_activations(ex.text, ident.layer, ident.neuron_idx, ident.pole)
        tokens = runner.encode(ex.text)["input_ids"][0].tolist()
        words = [runner.tokenizer.decode([i]) for i in tokens] if runner.tokenizer else None
        if words is not None and words != ex.tokens:
            report.fail(f"top example {n}: tokens differ from the record")
            continue
        values = strengths[0].tolist()
        report.compared += len(values)
        bad = [
            i
            for i, (a, b) in enumerate(zip(ex.activations, values, strict=False))
            if not tolerance.close(a, b, scale)
        ]
        if len(values) != len(ex.activations) or bad:
            where = f"token {bad[0]}" if bad else "length"
            report.fail(
                f"top example {n}: activations differ at {where} "
                f"(recorded {ex.activations[bad[0]] if bad else len(ex.activations)}, "
                f"reproduced {round(values[bad[0]], 6) if bad else len(values)})"
            )
        if not tolerance.close(ex.max_activation, max(values, default=0.0), scale):
            report.fail(f"top example {n}: max_activation differs")

    # 2. Counterfactuals as submitted.
    for n, pair in enumerate(sub.counterfactuals, 1):
        if pair.observed_delta is None or not scale:
            continue
        report.compared += 1
        delta = observed_delta(runner, ident, pair, scale)
        if not tolerance.close(pair.observed_delta, delta, 1.0):
            report.fail(
                f"counterfactual {n}: observed_delta recorded {pair.observed_delta}, "
                f"reproduced {delta}"
            )

    # 3. The review, under the record's own policy and reviewer.
    redo = evaluate_submission(
        sub, runner, probes_from_verdict(verdict), verdict.reviewer_agent_model, verdict.policy
    )
    _compare_verdicts(verdict, redo, tolerance, scale, report)
    return report


def _compare_environment(record: AuditRecord, now: RunEnvironment, report) -> None:
    revision, dtype = pinned_environment(record)
    if revision and now.model_revision and revision != now.model_revision:
        report.fail(f"model revision {now.model_revision} is not the pinned {revision}")
    if dtype not in {"auto", now.dtype}:
        report.fail(f"loaded {now.dtype} weights; the record pins {dtype}")
    then = record.verdict.environment
    if then is None:
        report.warnings.append("record predates environment pinning; device and versions unknown")
        return
    for name in ("device", "torch_version", "transformers_version"):
        if getattr(then, name) != getattr(now, name):
            report.warnings.append(
                f"{name}: recorded {getattr(then, name)}, now {getattr(now, name)}"
            )


def _compare_verdicts(old, new, tolerance: Tolerance, scale: float, report) -> None:
    if len(old.adversarial_test_results) != len(new.adversarial_test_results):
        report.fail("the record's test results don't match the probes and pairs it stores")
        return
    for n, (a, b) in enumerate(
        zip(old.adversarial_test_results, new.adversarial_test_results, strict=True), 1
    ):
        label = f"test {n} ({a.kind}: {a.prompt[:40]!r})"
        if (a.kind, a.prompt) != (b.kind, b.prompt):
            report.fail(f"{label}: does not match the stored probe")
            continue
        report.compared += 1
        if not tolerance.close(a.activation, b.activation, scale):
            report.fail(f"{label}: activation recorded {a.activation}, reproduced {b.activation}")
        if not tolerance.close(a.observed_delta, b.observed_delta, 1.0):
            report.fail(f"{label}: Δ recorded {a.observed_delta}, reproduced {b.observed_delta}")
        if a.passed != b.passed:
            report.fail(f"{label}: recorded passed={a.passed}, reproduced passed={b.passed}")

    if old.metrics is None or old.metrics.gates != new.metrics.gates:
        report.fail(
            f"gates recorded {old.metrics.gates if old.metrics else None}, "
            f"reproduced {new.metrics.gates}"
        )
    if not math.isclose(old.confidence_score, new.confidence_score, abs_tol=1e-6):
        report.fail(
            f"balanced accuracy recorded {old.confidence_score}, reproduced {new.confidence_score}"
        )
    if old.status != new.status:
        report.fail(f"status recorded {old.status}, reproduced {new.status}")

    oc, nc = old.causal_metrics, new.causal_metrics
    if (oc is None) != (nc is None) or (
        oc
        and nc
        and (
            abs(oc.sufficiency_score - nc.sufficiency_score) > 0.1
            or abs(oc.necessity_score - nc.necessity_score) > 0.1
        )
    ):
        report.warnings.append("causal metrics differ (recorded, not gated)")
