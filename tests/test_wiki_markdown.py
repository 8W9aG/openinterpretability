"""A lint of this project's rendered wiki templates, with scripts/check_markdown.py.

The template doesn't take this file: it renders this project's experiment pages (#6631).
"""

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from openinterpretability.builder.wiki import write_experiment_page, write_stop_page
from openinterpretability.core.draws import UnauditableDraw
from openinterpretability.core.schemas import AuditRecord, ExperimentProvenance

_SPEC = importlib.util.spec_from_file_location(
    "check_markdown", Path(__file__).parents[1] / "scripts" / "check_markdown.py"
)
check_markdown = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_markdown)


def test_rendered_experiment_and_index_templates_pass(tmp_path, record):
    """Two experiments on one unit also fill the history rows of the unit's page."""
    wiki = tmp_path / "wiki"
    write_experiment_page(record, wiki)
    probes = [
        t.model_copy(update={"passed": False, "measured": True})
        for t in record.verdict.adversarial_test_results
    ]
    update = {
        "status": "NEEDS_REVISION",
        "timestamp": record.verdict.timestamp + timedelta(hours=1),
        "adversarial_test_results": probes,
    }
    verdict = record.verdict.model_copy(update=update)
    second = write_experiment_page(AuditRecord(submission=record.submission, verdict=verdict), wiki)
    assert "Failed probes:" in second.read_text()
    pages = sorted(wiki.rglob("*.md"))
    assert len(pages) == 4  # home, unit page, two experiments
    problems = check_markdown.style_problems(pages)
    assert problems == [], "\n".join(problems)


@pytest.mark.parametrize(
    "fields",
    [
        {"outcome": "default_on", "active_share": 0.9},
        {"outcome": "dead", "proven_dead": True},
        {"outcome": "dead", "proven_dead": False},
    ],
)
def test_rendered_stop_template_passes(tmp_path, record, fields):
    wiki = tmp_path / "wiki"
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus="corpus.jsonl",
        corpus_sha256="ab" * 32,
        n_snippets=1,
        tokens_scanned=100,
        selection="manual",
        harness_version="test",
    )
    draw = UnauditableDraw(
        identifier=record.submission.identifier,
        provenance=provenance,
        timestamp=datetime(2026, 11, 2, 12, tzinfo=UTC),
        **fields,
    )
    page = write_stop_page(draw, wiki)
    assert check_markdown.style_problems([page]) == [], page.read_text()
