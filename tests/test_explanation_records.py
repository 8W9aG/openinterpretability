"""Explanation records and their pages (core/explanation_records.py, #3744 part B)."""

import re
from datetime import UTC, datetime
from types import SimpleNamespace

from openinterpretability.builder.compiler import compile_site
from openinterpretability.builder.explanation_pages import load_views, unit_href
from openinterpretability.core.explanation_records import (
    ExplainedUnit,
    ExplanationRecord,
    Narration,
    load_all,
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


def test_a_prompt_keeps_its_history(tmp_path):
    old = _write(tmp_path, _record(datetime(2026, 11, 16, tzinfo=UTC)))
    new = _write(tmp_path, _record(datetime(2026, 11, 17, tzinfo=UTC)))
    assert old.parent == new.parent and old != new  # one folder per prompt, a file per time
    (found,) = load_all(tmp_path).values()
    assert [p for p, _ in found] == [old, new]  # oldest first
    assert load_all(tmp_path / "missing") == {}


def test_an_explanation_is_stale_when_a_cited_units_status_changed(tmp_path):
    wiki = tmp_path / "wiki"
    _write(wiki, _record(datetime(2026, 11, 17, tzinfo=UTC)))
    (current,) = load_views(wiki, states={})
    assert not current.stale  # still unstudied
    verified = {
        SLUG: SimpleNamespace(
            record=SimpleNamespace(verdict=SimpleNamespace(status="VERIFIED")),
            standing=("VERIFIED", None),
        )
    }
    (view,) = load_views(wiki, states=verified)
    assert view.changed == [{"unit": "L10/N2931-", "then": "unstudied", "now": "VERIFIED"}]
    assert view.href == f"models/openai-community--gpt2/explanations/{view.record.prompt_id}.html"
    assert unit_href(SLUG) == "models/openai-community--gpt2/L10/N2931.html"
    assert unit_href(None) is None


def test_the_site_has_a_page_for_each_explanation(tmp_path, record):
    wiki = tmp_path / "wiki"
    _write(wiki, _record(datetime(2026, 11, 17, tzinfo=UTC)))
    views = load_views(wiki, states={})
    compile_site([record], tmp_path / "site", explanations=views)
    index = (tmp_path / "site" / "explanations" / "index.html").read_text()
    assert "John gave a drink to" in index and "current" in index
    page = (tmp_path / "site" / views[0].newest.href).read_text()
    assert "Mostly not known yet." in page and "L10/N2931-" in page and "C4" in page
    # No experiment in this site studies L10/N2931, so the unit has no page and no link
    # (#5727). The record fixture's own unit has a page.
    assert "models/openai-community--gpt2/L10/N2931.html" not in page
    linked = re.findall(r'href="([^"]*/L\d+/[NH]\d+\.html)"', page)
    here = (tmp_path / "site" / views[0].newest.href).parent
    assert not [h for h in linked if not (here / h).resolve().exists()]
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
    kept = [p for found in load_all(wiki).values() for p, _ in found]
    assert kept == [good]  # the bad one doesn't stop the rest
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
    assert f'href="../../models/{slug}/explanations/{mine.prompt_id}.html"' in section
    assert "John gave a drink to" in section and "openai-community/gpt2" not in section
    index = (tmp_path / "site" / "explanations" / "index.html").read_text()
    assert index.count("John gave a drink to") == 2  # the index lists both models'
    compile_site([record], tmp_path / "bare")
    bare = (tmp_path / "bare" / "models" / slug / "index.html").read_text()
    assert 'id="explanations"' not in bare


def _built(tmp_path, record, *records):
    wiki = tmp_path / "wiki"
    for r in records:
        _write(wiki, r)
    views = load_views(wiki, states={})
    compile_site([record], tmp_path / "site", explanations=views)
    return views


def test_a_prompt_page_lists_each_explanation_and_the_later_one_builds_on_the_earlier(
    tmp_path, record
):
    # The done-when of #4503: a 10% and a 20% explanation of one prompt.
    first = _record(datetime(2026, 11, 16, tzinfo=UTC)).model_copy(
        update={"target": 0.1, "prompt_coverage": 0.12}
    )
    second = _record(datetime(2026, 11, 17, tzinfo=UTC)).model_copy(
        update={"target": 0.2, "prompt_coverage": 0.21, "previous": first.id})  # fmt: skip
    (prompt,) = _built(tmp_path, record, first, second)
    assert [v.record.id for v in prompt.explanations] == [second.id, first.id]
    assert prompt.newest.previous.record.id == first.id and prompt.best_coverage == 0.21
    site = tmp_path / "site"
    page = (site / prompt.href).read_text()
    for view in prompt.explanations:  # the table links both, from the prompt's folder
        assert f'href="../../../{view.href}"' in page
        assert (site / view.href).exists()
    later = (site / prompt.newest.href).read_text()
    assert f'href="../../../../{prompt.explanations[1].href}"' in later  # names the 10% one
    assert "which reached 12%; this one reaches 21%" in later
    old = site / "explanations" / "openai-community--gpt2" / f"{first.prompt_id}.html"
    assert f"../../{prompt.href}" in old.read_text()  # the old page redirects


def test_the_index_groups_prompts_by_model_with_a_count_and_the_best_coverage(tmp_path, record):
    a = _record(datetime(2026, 11, 16, tzinfo=UTC)).model_copy(update={"prompt_coverage": 0.3})
    b = _record(datetime(2026, 11, 17, tzinfo=UTC)).model_copy(update={"prompt_coverage": 0.1})
    _built(tmp_path, record, a, b)
    index = (tmp_path / "site" / "explanations" / "index.html").read_text()
    assert index.count("John gave a drink to") == 1 and "openai-community/gpt2</span>" in index
    assert ">30%<" in index  # the best coverage, not the newest


def test_levels_ids_and_the_best_explanation():
    from openinterpretability.core.explanation_records import best

    when = datetime(2026, 11, 17, 5, 6, 7, tzinfo=UTC)
    rec = _record(when)
    assert rec.id == "20261117T050607Z" and rec.level == "long"
    short = rec.model_copy(update={"narration": rec.narration.model_copy(update={"sections": []})})
    assert short.level == "paragraph" and rec.model_copy(update={"narration": None}).level == (
        "mechanism")  # fmt: skip
    low = rec.model_copy(update={"prompt_coverage": 0.1})
    tie = rec.model_copy(update={"prompt_coverage": 0.1, "written": when.replace(hour=6)})
    assert best([low, tie]) is tie and best([]) is None
    assert best([rec.model_copy(update={"prompt_coverage": None})]) is None


def test_the_home_page_counts_each_explained_prompt_once(tmp_path, record):
    """Two runs on one prompt and one on another: the tile says 2, and links the index (#4574)."""
    wiki = tmp_path / "wiki"
    _write(wiki, _record(datetime(2026, 11, 17, tzinfo=UTC)))
    _write(wiki, _record(datetime(2026, 11, 18, tzinfo=UTC)))  # a second run, same prompt
    other = _record(datetime(2026, 11, 18, tzinfo=UTC)).model_copy(update={"prompt": "The sky is"})
    _write(wiki, other)
    views = load_views(wiki, states={})
    assert len(views) == 2
    compile_site([record], tmp_path / "site", explanations=views)
    home = (tmp_path / "site" / "index.html").read_text()
    tile = home[home.index("Explanations</a>") :]
    assert 'href="explanations/index.html"' in home and ">2</dd>" in tile[:400]
    compile_site([record], tmp_path / "bare")
    bare = (tmp_path / "bare" / "index.html").read_text()
    assert "Explanations</dt>" in bare and 'href="explanations/index.html"' not in bare
