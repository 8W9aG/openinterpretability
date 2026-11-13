"""Tests for the model-free verdict recomputation in openinterp/verdict_regression.py."""

import pytest

from openinterp.agents.reviewer import decide, review_metrics
from openinterp.builder.wiki import write_experiment_page
from openinterp.checks import check_registry
from openinterp.core.schemas import AdversarialTestResult, AuditRecord, ReviewPolicy
from openinterp.verdict_regression import verdict_drift


def _results():
    negatives = [
        AdversarialTestResult(
            kind="hard_negative",
            prompt=f"plain sentence number {w}",
            passed=True,
            expected_active=False,
            activation=0.0,
        )
        for w in ("one", "two", "three", "four", "five", "six", "seven")
    ]
    holdouts = [
        AdversarialTestResult(
            kind="positive_holdout",
            prompt=f"a quoted {w} phrase",
            passed=w != "gamma",
            expected_active=True,
            activation=1.0,
        )
        for w in ("alpha", "beta", "gamma", "delta")
    ]
    return negatives + holdouts


@pytest.fixture
def judged(submission, verdict) -> AuditRecord:
    """A record whose stored metrics and status are what the current code gives."""
    policy = ReviewPolicy()
    results = _results()
    metrics = review_metrics(results, policy, independent=True, baseline=("never active", 0.5))
    update = {
        "adversarial_test_results": results,
        "policy": policy,
        "metrics": metrics,
        "status": decide(metrics, policy),
        "confidence_score": metrics.balanced_accuracy,
    }
    return AuditRecord(submission=submission, verdict=verdict.model_copy(update=update))


def _with(record, **update):
    return record.model_copy(update={"verdict": record.verdict.model_copy(update=update)})


def test_matching_record_has_no_drift(judged):
    assert verdict_drift(judged) is None


def test_changed_status_and_gate_are_reported(judged):
    gates = {
        **judged.verdict.metrics.gates,
        "sensitivity": not judged.verdict.metrics.gates["sensitivity"],
    }
    metrics = judged.verdict.metrics.model_copy(update={"gates": gates})
    message = verdict_drift(_with(judged, status="REJECTED", metrics=metrics))
    assert "status REJECTED -> " in message and "gates sensitivity" in message


def test_changed_balanced_accuracy_is_reported(judged):
    metrics = judged.verdict.metrics.model_copy(update={"balanced_accuracy": 0.123})
    assert "balanced accuracy 0.123 -> " in verdict_drift(_with(judged, metrics=metrics))


def test_older_protocol_is_not_checked(judged):
    old = judged.verdict.policy.model_copy(update={"protocol_version": "0.7"})
    assert verdict_drift(_with(judged, status="REJECTED", policy=old)) is None


def test_record_without_metrics_is_not_checked(judged):
    assert verdict_drift(_with(judged, metrics=None, status="REJECTED")) is None


def test_check_registry_reports_drift(tmp_path, monkeypatch, judged):
    monkeypatch.chdir(tmp_path)
    write_experiment_page(_with(judged, status="REJECTED"), tmp_path / "wiki")
    report = check_registry(tmp_path / "wiki")
    assert any("different verdict" in p.message for p in report.problems)
