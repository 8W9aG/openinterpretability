from datetime import timedelta

from openinterp.builder.wiki import (
    _read_front_matter,
    backfill,
    experiment_id,
    load_entries,
    write_experiment_page,
)
from openinterp.core.schemas import AuditRecord, ReviewMetrics, ReviewPolicy


def _failed_record(record):
    metrics = ReviewMetrics(
        specificity=1.0,
        sensitivity=0.4,
        balanced_accuracy=0.7,
        counterfactual_reproduction=0.25,
        lookahead_consistent=True,
        independent_review=False,
        gates={"specificity": True, "sensitivity": False, "independent_review": False},
    )
    verdict = record.verdict.model_copy(
        update={
            "status": "NEEDS_REVISION",
            "metrics": metrics,
            "policy": ReviewPolicy(),
            "timestamp": record.verdict.timestamp + timedelta(hours=1),
        }
    )
    return AuditRecord(submission=record.submission, verdict=verdict)


def test_writes_experiment_page_and_indexes(tmp_path, record):
    wiki = tmp_path / "wiki"

    page = write_experiment_page(record, wiki)
    assert page == wiki / "experiments" / record.submission.identifier.slug / "20261001T120000Z.md"
    text = page.read_text()
    meta = _read_front_matter(page)
    assert meta["status"] == "VERIFIED" and meta["layer"] == 3
    assert "Fires on the closing quote of reported speech." in text
    assert "⟦" in text  # peak token marked in examples
    assert "(20261001T120000Z.json)" in text  # links its own archived record
    assert "## Discussion" in text

    home = (wiki / "README.md").read_text()
    assert "L3/N117+" in home and "VERIFIED: 1" in home
    neuron = (page.parent / "README.md").read_text()
    assert "20261001T120000Z.md" in neuron


def test_pages_are_write_once(tmp_path, record):
    wiki = tmp_path / "wiki"
    page = write_experiment_page(record, wiki)
    page.write_text(page.read_text() + "\nMy notes\n")
    assert write_experiment_page(record, wiki) is None
    assert "My notes" in page.read_text()


def test_history_and_next_steps(tmp_path, record):
    wiki = tmp_path / "wiki"
    write_experiment_page(record, wiki)
    failed = _failed_record(record)
    page = write_experiment_page(failed, wiki, kind="reevaluation", previous_status="VERIFIED")

    text = page.read_text()
    assert "previous verdict VERIFIED" in text
    assert "does not generalise" in text  # sensitivity advice
    assert "different reviewer model" in text  # independence advice
    assert "too broad" not in text  # specificity passed

    entries = load_entries(wiki)
    assert [e.meta["experiment_id"] for e in entries] == [
        experiment_id(failed),
        experiment_id(record),
    ]
    neuron = (page.parent / "README.md").read_text()
    assert "**Latest status:** NEEDS_REVISION" in neuron


def test_table_cells_escaped(tmp_path, record):
    sub = record.submission.model_copy(
        update={"natural_language_explanation": "Fires on a | pipe\nand newline"}
    )
    page = write_experiment_page(AuditRecord(submission=sub, verdict=record.verdict), tmp_path)
    home = (tmp_path / "README.md").read_text()
    row = next(line for line in home.splitlines() if "N117" in line)
    assert "a \\| pipe ⏎ and newline" in row
    assert page.exists()


def test_backfill_skips_unreviewed(tmp_path, record, submission):
    written = backfill([record, AuditRecord(submission=submission)], tmp_path / "wiki")
    assert len(written) == 1
    assert backfill([record], tmp_path / "wiki") == []
