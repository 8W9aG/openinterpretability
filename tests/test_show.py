import os

import pytest
from click.testing import CliRunner

from openinterpretability.builder.wiki import archive_path, write_experiment_page
from openinterpretability.cli import cli, quiet_libraries
from openinterpretability.core.examples import OppositePole
from openinterpretability.core.explanation import ExplanationScores, JudgeScore
from openinterpretability.core.fence import fence
from openinterpretability.core.partners import CorrelatedNeuron
from openinterpretability.core.schemas import (
    AdversarialTestResult,
    ExperimentProvenance,
    ReviewMetrics,
    ReviewPolicy,
    ScanEvidence,
    VocabularyLens,
)
from openinterpretability.show import show, show_evidence, show_record


def _show(*args):
    result = CliRunner().invoke(cli, ["show", *map(str, args)])
    assert result.exit_code == 0, result.output
    return result.output


def test_show_record_lists_failed_probes_and_discussion(tmp_path, record):
    failed = AdversarialTestResult(
        kind="hard_negative",
        prompt="He said nothing at all, " * 20,
        passed=False,
        expected_active=False,
        activation=1.5,
    )
    verdict = record.verdict.model_copy(
        update={"adversarial_test_results": [*record.verdict.adversarial_test_results, failed]}
    )
    record = record.model_copy(update={"verdict": verdict})
    page = write_experiment_page(record, tmp_path / "wiki")
    page.write_text(page.read_text() + "\nThe quote mark is the trigger.\n")
    path = archive_path(tmp_path / "wiki", record)

    text = _show(path)
    assert "VERIFIED 0.86" in text
    # Text a PR could write is fenced as data for the agents that read this.
    assert fence("Fires on the closing quote of reported speech.") in text
    assert "Failed probes: 1 of 2" in text
    assert "✗ hard negative: 1.500, expected inactive: He said nothing" in text
    assert "…" in text and "He said nothing at all, " * 20 not in text  # cut
    assert "Discussion:\n" + fence("The quote mark is the trigger.") in text
    assert "Free-form notes" not in text  # the template comment is dropped
    assert ("He said nothing at all, " * 19) in _show(path, "--full")

    unit = _show(path.parent)
    assert unit.startswith(f"1 experiment(s) in {path.parent}") and "VERIFIED 0.86" in unit


def test_show_evidence_and_submission(tmp_path, submission):
    ev = ScanEvidence(
        identifier=submission.identifier,
        top_examples=submission.top_examples,
        activation_histogram=submission.activation_histogram,
    )
    path = tmp_path / "ev.json"
    path.write_text(ev.model_dump_json())
    text = _show(path)
    assert "scan evidence" in text and "Max activation 2.500" in text
    assert '"Hi,["] he said' in text

    path.write_text(submission.model_dump_json())
    assert "submission by openai/gpt-test" in _show(path)


def test_show_rejects_other_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}")
    result = CliRunner().invoke(cli, ["show", str(path)])
    assert result.exit_code == 2 and "not a record" in result.output


def test_quiet_libraries(monkeypatch):
    for name in ("HF_HUB_DISABLE_PROGRESS_BARS", "TRANSFORMERS_VERBOSITY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("warnings.filters", [])
    quiet_libraries(quiet=True)
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"


def _provenance(**update):
    fields = dict(
        dtype="float32", corpus="tiny", n_snippets=4, tokens_scanned=12345,
        selection="random", seed=7, harness_version="0",
    )  # fmt: skip
    return ExperimentProvenance(**{**fields, **update})


def test_show_unit_rejects_a_directory_without_records(tmp_path):
    with pytest.raises(ValueError, match="no experiment records"):
        show(tmp_path)


def test_show_record_without_a_verdict_says_not_reviewed(record):
    text = show_record(record.model_copy(update={"verdict": None}))
    assert text.startswith("Qwen")
    assert " · not reviewed\n== " in text and "submission by openai/gpt-test" in text


def test_show_record_lists_gates_rates_scores_notes_and_counterfactual_failures(tmp_path, record):
    metrics = ReviewMetrics(
        specificity=0.5, sensitivity=None, balanced_accuracy=0.75,
        counterfactual_reproduction=1.0, lookahead_consistent=True,
        independent_review=True, gates={"specificity": False, "sensitivity": True},
        confound_baseline_accuracy=0.4, confound_baseline="token",
    )  # fmt: skip
    judge = JudgeScore(
        balanced_accuracy=0.9, true_positive_rate=1.0, false_positive_rate=0.2,
        recall_by_quantile=[1.0],
    )  # fmt: skip
    scores = ExplanationScores(
        slug="s", explanation="x", threshold=0.6, window_tokens=32, quantiles=1,
        quantile_edges=[], seed=3, corpus="c", candidate_windows=2, active_windows=1,
        detection=judge,
    )  # fmt: skip
    cf = AdversarialTestResult(
        kind="counterfactual", prompt="Hello, she said.", passed=False,
        expected_delta=0.8, observed_delta=None,
    )  # fmt: skip
    verdict = record.verdict.model_copy(
        update={
            "metrics": metrics, "explanation_scores": scores, "causal_metrics": None,
            "policy": ReviewPolicy(), "notes": "Check   the quote.",
            "adversarial_test_results": [cf],
        }
    )  # fmt: skip
    sub = record.submission.model_copy(update={"provenance": _provenance()})
    path = tmp_path / "rec.json"  # no page beside it, so no Discussion
    text = show_record(record.model_copy(update={"submission": sub, "verdict": verdict}), path)

    assert "· selection random (seed 7)" in text
    assert "Failed gates: specificity" in text
    assert (
        "specificity 0.50 · balanced accuracy 0.75 · counterfactuals 1.00"
        " · best baseline (token) 0.40"
    ) in text
    assert "Explanation scores: detection 0.90" in text and "fuzzing" not in text
    assert "Causal:" not in text and "Discussion:" not in text
    assert "Failed probes: 1 of 1 (active line 0.625):" in text
    assert "✗ counterfactual: Δ n/a vs expected 0.800: Hello, she said." in text
    assert "Notes:\n" + fence("Check the quote.") in text


def test_show_record_with_no_probes_and_all_gates_passed(tmp_path, record):
    metrics = ReviewMetrics(
        specificity=None, sensitivity=None, balanced_accuracy=1.0,
        counterfactual_reproduction=None, lookahead_consistent=True,
        independent_review=True, gates={"specificity": True},
    )  # fmt: skip
    verdict = record.verdict.model_copy(update={"metrics": metrics, "adversarial_test_results": []})
    page = tmp_path / "rec.md"
    page.write_text("# A page with no Discussion section\n")
    text = show_record(record.model_copy(update={"verdict": verdict}), tmp_path / "rec.json")
    assert "Failed gates: none\nbalanced accuracy 1.00\n" in text
    assert "Failed probes" not in text and "Discussion:" not in text


def test_show_evidence_reports_provenance_lens_partners_and_opposite_pole(submission):
    lens = VocabularyLens(
        logit_promoted=[f"t{i}" for i in range(12)], logit_suppressed=["x"],
        jacobian_promoted=[], jacobian_suppressed=[], overlap=0.0, direction_cosine=0.5,
        logit_kurtosis=1.0, jacobian_kurtosis=1.0, prompts=2,
    )  # fmt: skip
    partner = CorrelatedNeuron(
        neuron_idx=9, pole="negative", coactivation_lift=3.0, coactivation_lift_lower=2.04,
        coactive_tokens=6, spearman=0.3, pearson=0.2,
    )  # fmt: skip
    opposite = OppositePole(pole="negative", top_examples=submission.top_examples)
    ev = ScanEvidence(
        identifier=submission.identifier, top_examples=submission.top_examples,
        first_token_floor=True, provenance=_provenance(snippet_tokens=64),
        vocabulary_lens=lens, correlated_neurons=[partner], opposite_pole=opposite,
    )  # fmt: skip
    text = show_evidence(ev)
    assert (
        "Corpus tiny: 4 snippets, 12,345 tokens, 64-token windows · selection random (seed 7)"
    ) in text
    # No histogram, so no active share, but the first-token floor is named.
    assert "Max activation 2.500 · first-token floor\n" in text
    assert "Logit lens promotes: 't0'" in text and "(+2)" in text
    assert "Logit lens suppresses: 'x'" in text
    assert "Co-activating poles: N9− (lift ≥ 2.0)" in text
    assert 'Opposite pole (negative) peak 2.500: "Hi,["] he said' in text


def test_show_evidence_without_provenance_or_seed(submission):
    ev = ScanEvidence(
        identifier=submission.identifier, top_examples=[],
        provenance=_provenance(seed=None),
    )  # fmt: skip
    text = show_evidence(ev)
    assert "Corpus tiny: 4 snippets, 12,345 tokens\n" in text
    assert "Max activation 0.000\nTop examples (0 of 0)" in text
