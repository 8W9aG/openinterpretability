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


def test_records_only_hold_strict_names_and_slugs():
    import pytest
    from pydantic import ValidationError

    good = _record(datetime(2026, 11, 17, tzinfo=UTC))
    for bad in ("gpt2; x", "a b", "-m", "org/name/extra"):
        with pytest.raises(ValidationError):
            ExplanationRecord.model_validate(good.model_dump() | {"model_name": bad})
    for bad in ("m/L1/N1-positive;x", "m/L1/N1", "m/L1/H2 x", "../L1/H2"):
        with pytest.raises(ValidationError):
            ExplainedUnit(unit="u", slug=bad, kind="pole", share=0, attribution=0, status="x")
    assert unit_href("m/L1/N1-positive;x") is None  # matched in full


def test_an_invalid_record_is_skipped_and_the_check_rejects_it(tmp_path, monkeypatch, capsys):
    from openinterpretability import planner
    from openinterpretability.checks import CheckReport
    from openinterpretability.core.explanation_records import check_explanations

    wiki = tmp_path / "wiki"
    good = _write(wiki, _record(datetime(2026, 11, 17, tzinfo=UTC)))
    bad = wiki / "explanations" / "x" / "y" / "z.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{}")
    assert [p for p, _ in load_latest(wiki)] == [good]  # the bad one doesn't stop the rest
    assert "skipping an invalid explanation record" in capsys.readouterr().err
    monkeypatch.setattr(planner, "configured_models", lambda: ("openai-community/gpt2",))
    monkeypatch.setattr(planner, "signed_models", lambda: ())
    report = CheckReport()
    check_explanations(wiki, report)
    assert [(p.path, "valid" in p.message) for p in report.problems] == [(bad, True)]
    monkeypatch.setattr(planner, "configured_models", lambda: ("other/model",))
    report = CheckReport()
    check_explanations(wiki, report)
    assert any("not a configured model" in p.message for p in report.problems)


def test_each_model_page_lists_its_explanations(tmp_path, record):
    # The per-model Explanations table (#4206), with rows from the shared macro.
    model = record.submission.identifier.model_name
    wiki = tmp_path / "wiki"
    mine = _record(datetime(2026, 11, 17, tzinfo=UTC)).model_copy(update={"model_name": model})
    _write(wiki, mine)
    _write(wiki, _record(datetime(2026, 11, 16, tzinfo=UTC)))  # another model's
    views = load_views(wiki, states={})
    compile_site([record], tmp_path / "site", explanations=views)
    slug = record.submission.identifier.model_slug
    page = (tmp_path / "site" / "models" / slug / "index.html").read_text()
    section = page[page.index('id="explanations"') :]
    assert f'href="../../explanations/{slug}/{mine.prompt_id}.html"' in section
    assert "John gave a drink to" in section and "openai-community/gpt2" not in section
    index = (tmp_path / "site" / "explanations" / "index.html").read_text()
    assert index.count("John gave a drink to") == 2  # the index lists both models'
    compile_site([record], tmp_path / "bare")
    bare = (tmp_path / "bare" / "models" / slug / "index.html").read_text()
    assert 'id="explanations"' not in bare
