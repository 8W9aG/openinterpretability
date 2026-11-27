"""What the planner knows of each unit, from its records and their errata (#6810).

These tests are fast, so ``[tool.mutmut]`` selects them for ``planner_states.py``.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from openinterpretability import planner_states
from openinterpretability.builder import wiki
from openinterpretability.core.errata import Erratum, record_ref
from openinterpretability.core.schemas import AuditRecord


def later(record: AuditRecord, hours: int, reviewer: str | None = None) -> AuditRecord:
    update = {"timestamp": record.verdict.timestamp + timedelta(hours=hours)}
    if reviewer:
        update["reviewer_agent_model"] = reviewer
    return record.model_copy(update={"verdict": record.verdict.model_copy(update=update)})


def set_aside(*records: AuditRecord) -> Erratum:
    return Erratum.model_validate_json(json.dumps({
        "id": "bug", "date": "2026-11-13", "title": "A bug", "bug": "What it did.",
        "fixed_in": "abcdef1", "set_aside": True, "records": [record_ref(r) for r in records],
    }))  # fmt: skip


def test_the_planner_skips_records_whose_evidence_is_missing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(wiki, "load_archive_paths", lambda d, **kw: calls.append((d, kw)) or [])
    assert planner_states.load_states(tmp_path) == {}
    assert calls == [(tmp_path, {"skip_missing": True})]


def test_a_set_aside_record_after_a_counted_one_is_not_the_units_record(record):
    newer = later(record, 1)
    pairs = [(Path("old.json"), record), (Path("new.json"), newer)]
    state = planner_states.states_of(pairs, [set_aside(newer)])[record.submission.identifier.slug]
    assert (state.record, state.record_path) == (record, Path("old.json"))
    assert state.experiments == 2 and state.last == newer.verdict.timestamp


def test_a_unit_whose_records_are_all_set_aside_keeps_its_newest(record):
    newer = later(record, 1)
    pairs = [(Path("old.json"), record), (Path("new.json"), newer)]
    state = planner_states.states_of(pairs, [set_aside(record, newer)])
    assert state[record.submission.identifier.slug].record_path == Path("new.json")


def test_independent_reviews_add_up(record):
    other = later(record, 1, reviewer="openai/another-model")
    assert not record.self_reviewed and not other.self_reviewed
    pairs = [(Path("a.json"), record), (Path("b.json"), other)]
    state = planner_states.states_of(pairs, [])[record.submission.identifier.slug]
    assert state.independent_reviews == 2
    assert len(state.reviewers) == 2
