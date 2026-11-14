"""Takedowns hide a passage everywhere the build publishes it (issue #3270)."""

import json
from datetime import date
from pathlib import Path

import pytest

from openinterp.builder import pii
from openinterp.builder.wiki import write_experiment_page
from openinterp.checks import check_registry
from openinterp.core import takedowns as td
from openinterp.signing import sign_record
from openinterp.takedown_checks import page_redaction_ok
from tests.test_errata import git, repo  # noqa: F401  (fixture)

PASSAGE = "Fires on the closing quote of reported speech."  # the fixture's hypothesis


@pytest.fixture
def active():
    """Turn on a takedown of PASSAGE for one test, with fresh span caches."""
    pii.find_spans.cache_clear()
    td.activate([_takedown()])
    yield
    td.activate([])
    pii.find_spans.cache_clear()


def _takedown(id_="t-1"):
    return td.Takedown(
        id=id_, date=date(2026, 11, 14), reason="test", shingles=td.shingle_hashes(PASSAGE)
    )


def test_a_passage_needs_a_whole_shingle_and_is_named_only_by_hashes():
    with pytest.raises(ValueError, match="at least 8 words"):
        td.shingle_hashes("too short to name")
    line = td.to_line(_takedown())
    assert "quote" not in line and len(json.loads(line)["shingles"]) == 1


def test_matching_ignores_case_and_punctuation_and_marks_only_the_passage():
    hashes = td.all_hashes([_takedown()])
    text = 'Intro. FIRES on the "closing" quote, of reported speech! Outro.'
    (span,) = td.covered(text, hashes)
    assert text[span[0] : span[1]].startswith("FIRES") and text[span[1] :] == "! Outro."
    assert td.redact(text, hashes, "[x]") == "Intro. [x]! Outro."
    assert td.covered("Fires on the closing quote of indirect speech.", hashes) == ()


def test_the_build_removes_the_passage_from_text_and_tokens(active, record):
    assert pii.redact_text(f"He wrote: {PASSAGE}") == f"He wrote: {pii.REMOVED_TEXT}."
    tokens = ["A ", *[w + " " for w in PASSAGE.split()], "end"]
    redacted = pii.redact_tokens(tokens)
    assert redacted[0] == "A " and redacted[-1] == "end"
    assert set(redacted[1:-1]) == {pii.REMOVED_TOKEN}
    published = pii.scrub_record(record)
    assert PASSAGE not in published.model_dump_json()
    assert record.submission.natural_language_explanation == PASSAGE  # on disk, unchanged


def test_a_page_may_change_only_by_the_takedown(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "takedowns.jsonl").write_text(td.to_line(_takedown()))
    before = f"# Page\n\n> {PASSAGE}\n\nVerdict: VERIFIED\n"
    after = td.redact(before, td.all_hashes([_takedown()]), pii.REMOVED_TEXT)
    assert page_redaction_ok(before, after, wiki)
    assert not page_redaction_ok(before, before, wiki)  # no change is not a redaction
    assert not page_redaction_ok(before, after.replace("VERIFIED", "REJECTED"), wiki)
    assert not page_redaction_ok(before, before.replace("Verdict: VERIFIED\n", ""), wiki)


def _messages(report):
    return " | ".join(p.message for p in report.problems)


def test_check_needs_a_signed_append_only_file_and_accepts_the_redacted_page(repo):  # noqa: F811
    path = td.takedowns_path("wiki")
    path.write_text(td.to_line(_takedown()))
    page = next(Path("wiki").rglob("2*.md"))
    head, sep, rest = page.read_text().partition("## Discussion")
    page.write_text(td.redact(head, td.all_hashes([_takedown()]), pii.REMOVED_TEXT) + sep + rest)
    assert PASSAGE not in page.read_text()
    git("add", ".")
    git("commit", "-qm", "takedown")
    assert "takedown file is unsigned" in _messages(check_registry(base_ref="main~1"))

    sign_record(path, repo["key"])
    git("add", ".")
    git("commit", "-qm", "sign")
    report = check_registry(base_ref="main~2")
    assert report.ok, _messages(report)

    path.write_text(td.to_line(_takedown("t-2")))  # rewrites the first line
    sign_record(path, repo["key"])
    git("commit", "-qam", "rewrite")
    assert "append-only" in _messages(check_registry(base_ref="main~1"))


def test_check_rejects_a_bad_line_and_a_repeated_id(repo):  # noqa: F811
    path = td.takedowns_path("wiki")
    path.write_text('{"id": "x"}\n')
    assert "invalid takedown: line 1" in _messages(check_registry())
    path.write_text(td.to_line(_takedown()) * 2)
    assert "used more than once" in _messages(check_registry())


def test_the_script_appends_the_takedown_and_redacts_the_pages(tmp_path, record, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import takedown

    wiki = tmp_path / "wiki"
    write_experiment_page(record, wiki)
    passage = tmp_path / "passage.txt"
    passage.write_text(PASSAGE)
    args = ["takedown.py", "--passage", str(passage), "--id", "t-1", "--reason", "test"]
    monkeypatch.setattr(sys, "argv", [*args, "--wiki-dir", str(wiki)])
    assert takedown.main() == 0
    (line,) = td.load_takedowns(wiki)
    assert line.id == "t-1" and line.shingles == td.shingle_hashes(PASSAGE)
    page = next(wiki.rglob("2*.md")).read_text()
    assert PASSAGE not in page.split("## Discussion")[0] and pii.REMOVED_TEXT in page
    record_file = next(wiki.rglob("2*.json")).read_text()
    assert PASSAGE in record_file  # the signed record doesn't change
