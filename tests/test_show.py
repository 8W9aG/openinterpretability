import os

from click.testing import CliRunner

from openinterp.builder.wiki import archive_path, write_experiment_page
from openinterp.cli import cli, quiet_libraries
from openinterp.core.schemas import AdversarialTestResult, ScanEvidence


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
    assert "Hypothesis: Fires on the closing quote of reported speech." in text
    assert "Failed probes: 1 of 2" in text
    assert "✗ hard negative: 1.500, expected inactive: He said nothing" in text
    assert "…" in text and "He said nothing at all, " * 20 not in text  # cut
    assert "Discussion: The quote mark is the trigger." in text
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
