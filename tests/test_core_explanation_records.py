"""Explanation records: coverage details, the caution flag, and files the check refuses."""

import shutil
from datetime import UTC, datetime

from openinterpretability import planner
from openinterpretability.checks import CheckReport
from openinterpretability.core.explanation_records import (
    CoverageDetail,
    ExplanationRecord,
    check_explanations,
)
from tests.test_explanation_records import _record, _write

WHEN = datetime(2026, 11, 17, tzinfo=UTC)


def test_a_coverage_result_without_its_random_baseline_has_no_detail():
    assert CoverageDetail.of({"coverage": 0.4}) is None
    assert ExplanationRecord.detail({}) is None


def test_no_measured_coverage_is_no_caution():
    record = _record(WHEN).model_copy(update={"prompt_coverage": None})
    assert record.coverage_caution is False
    assert _record(WHEN).coverage_caution is True  # its share is under the caution share


def test_the_check_refuses_other_files_and_records_at_the_wrong_path(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "configured_models", lambda: ("openai-community/gpt2",))
    monkeypatch.setattr(planner, "signed_models", lambda: ())
    wiki = tmp_path / "wiki"
    good = _write(wiki, _record(WHEN))
    note = good.parent / "notes.txt"
    note.write_text("x")
    moved = good.parent / "20200101T000000Z.json"
    shutil.move(good, moved)
    report = CheckReport()
    check_explanations(wiki, report)
    problems = {p.path: p.message for p in report.problems}
    assert problems[note] == "only explanation records (.json) belong in wiki/explanations/"
    assert problems[moved] == f"should be at {good}"
