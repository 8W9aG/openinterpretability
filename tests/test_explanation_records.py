"""Explanation records and their pages (core/explanation_records.py, #3744 part B)."""

from datetime import UTC, datetime
from types import SimpleNamespace

from openinterpretability.builder.compiler import compile_site
from openinterpretability.builder.explanation_pages import load_views, unit_href
from openinterpretability.core.explanation_records import (
    ExplainedUnit,
    ExplanationRecord,
    Narration,
    load_latest,
    record_path,
)

SLUG = "openai-community--gpt2/L10/N2931-negative"


def _record(when, status="unstudied", answer=" Mary"):
    return ExplanationRecord(
        model_name="openai-community/gpt2", prompt="John gave a drink to", position=-1,
        answer=answer, explained_share=0.003, blocked_share=0.006, prompt_coverage=0.0,
        random_coverage=0.0, planned_experiments=54483, package_version="0.0.21", written=when,
        units=[ExplainedUnit(unit="L10/N2931-", slug=SLUG, kind="pole", share=0.008,
                             attribution=14.8, status=status),
               ExplainedUnit(unit="C4", kind="circuit", share=0.0003, attribution=-0.5,
                             status="NEEDS_REVISION")],
        narration=Narration(sections=[{"title": "Step", "text": "L10/N2931- carries it."}],
                            paragraph="Mostly not known yet.", narrator_model="external/x"),
    )  # fmt: skip


def _write(wiki, record):
    path = record_path(wiki, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json())
    return path


def test_a_prompt_keeps_its_history_and_the_newest_counts(tmp_path):
    old = _write(tmp_path, _record(datetime(2026, 11, 16, tzinfo=UTC)))
    new = _write(tmp_path, _record(datetime(2026, 11, 17, tzinfo=UTC)))
    assert old.parent == new.parent and old != new  # one folder per prompt, a file per time
    ((path, latest),) = load_latest(tmp_path)
    assert path == new and latest.written.day == 17
    assert load_latest(tmp_path / "missing") == []


def test_an_explanation_is_stale_when_a_cited_units_status_changed(tmp_path):
    wiki = tmp_path / "wiki"
    _write(wiki, _record(datetime(2026, 11, 17, tzinfo=UTC)))
    (current,) = load_views(wiki, states={})
    assert not current.stale  # still unstudied
    verified = {
        SLUG: SimpleNamespace(record=SimpleNamespace(verdict=SimpleNamespace(status="VERIFIED")))
    }
    (view,) = load_views(wiki, states=verified)
    assert view.changed == [{"unit": "L10/N2931-", "then": "unstudied", "now": "VERIFIED"}]
    assert view.href.startswith("explanations/openai-community--gpt2/") and view.href.endswith(
        ".html"
    )
    assert unit_href(SLUG) == "models/openai-community--gpt2/L10/N2931.html"
    assert unit_href(None) is None


def test_the_site_has_a_page_for_each_explanation(tmp_path, record):
    wiki = tmp_path / "wiki"
    _write(wiki, _record(datetime(2026, 11, 17, tzinfo=UTC)))
    views = load_views(wiki, states={})
    compile_site([record], tmp_path / "site", explanations=views)
    index = (tmp_path / "site" / "explanations" / "index.html").read_text()
    assert "John gave a drink to" in index and "current" in index
    page = (tmp_path / "site" / views[0].href).read_text()
    assert "Mostly not known yet." in page and "L10/N2931-" in page and "C4" in page
    assert 'href="../../models/openai-community--gpt2/L10/N2931.html"' in page
    research = (tmp_path / "site" / "research" / "index.html").read_text()
    assert "explanations/index.html" in research
    compile_site([record], tmp_path / "bare")
    assert not (tmp_path / "bare" / "explanations").exists()  # none recorded: no pages
