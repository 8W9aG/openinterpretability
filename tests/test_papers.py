"""The references and the structure of the papers in papers/."""

import re
from pathlib import Path

import pytest

PAPERS = Path(__file__).parents[1] / "papers"
WRITTEN = sorted(path.parent.name for path in PAPERS.glob("*/results.jsonl"))
CITATION = re.compile(
    r"((?:[A-Z][\w-]+ )*[A-Z][\w-]+)"
    r"(?: et al\.| (?:&|and) [A-Z][\w ]*?[a-z])?(?:, | \()((?:19|20)\d{2})\b"
)
REFERENCE = re.compile(r"^- ([^,]+), .*?\(((?:19|20)\d{2})\)\.(.*)$", re.M)


def _same(cited: str, listed: str) -> bool:
    """True when a citation's name and a reference's name are the same surname."""
    return cited.endswith(listed) or listed.endswith(cited)


def _sections(paper: str) -> tuple[str, str]:
    """Return a paper's body and its reference list."""
    text = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    body, _, references = text.partition("\n## References\n")
    return body, references


@pytest.mark.parametrize("paper", WRITTEN)
def test_paper_has_references_with_links(paper):
    _, references = _sections(paper)
    entries = REFERENCE.findall(references)
    assert entries
    assert all("](" in rest for _, _, rest in entries)


@pytest.mark.parametrize("paper", WRITTEN)
def test_each_citation_has_a_reference(paper):
    body, references = _sections(paper)
    listed = [(name, year) for name, year, _ in REFERENCE.findall(references)]
    for name, year in CITATION.findall(body):
        # Either side may hold a particle that the other lacks: "van der Wal" and "Wal".
        assert any(_same(name, n) and year == y for n, y in listed), f"{name}, {year}"


@pytest.mark.parametrize("paper", WRITTEN)
def test_each_reference_is_cited(paper):
    body, references = _sections(paper)
    cited = CITATION.findall(body)
    for name, year, _ in REFERENCE.findall(references):
        assert any(_same(c, name) and y == year for c, y in cited), f"{name}, {year}"


# APA 7, as the write-paper skill sets it. List every author, or the first 19, an
# ellipsis and the last. Put a comma before the final "&". Then "(Year).", and the title
# in italics.
AUTHOR = r"[^,()]+, (?:[A-Z]\.(?:[ -][A-Z]\.)*|[A-Z][\w.]*\.?)"
APA = re.compile(
    rf"^- {AUTHOR}(?:, {AUTHOR})*(?:,? (?:& |\. \. \. ){AUTHOR})? \((?:19|20)\d{{2}}\)\. \*[^*]+\*"
)


@pytest.mark.parametrize("paper", WRITTEN)
def test_references_follow_apa(paper):
    _, references = _sections(paper)
    entries = [line for line in references.splitlines() if REFERENCE.match(line)]
    for line in entries:
        assert "et al." not in line, line
        authors = line.split(" (", 1)[0]
        assert " & " not in authors or ", & " in authors, line
        assert APA.match(line), line


# The sections of a paper, in order (write-paper skill, #4202). An experiment section may
# have any title, so only these fixed sections are checked. Others may stand between them.
SECTIONS = ("Introduction", "Related work", "Method", "Discussion", "Limitations",
            "Conclusion", "References", "Appendices")  # fmt: skip
# Papers written before #4202, each with the issue that rewrites it. Only shrink this list.
OLD_FORM = {
    "explanations-across-models": 4246, "native-vs-learned": 4247,
    "spd-input-projections": 4248, "syntax-positive-controls": 4249, "why-neurons-rest": 4250,
}  # fmt: skip
NEW_FORM = [paper for paper in WRITTEN if paper not in OLD_FORM]


def _headings(paper: str) -> list[str]:
    text = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    return [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]


@pytest.mark.parametrize("paper", NEW_FORM)
def test_paper_has_the_sections_in_order(paper):
    headings = _headings(paper)
    found = [h for h in headings if h in SECTIONS]
    assert found == list(SECTIONS), f"{paper}: sections {found}, want {list(SECTIONS)}"
    text = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    intro = text.split("\n## Introduction\n", 1)[1].split("\n## ", 1)[0]
    assert "research-questions.md" in intro, f"{paper}: link the research question"
    assert "contribut" in intro.lower(), f"{paper}: list the contributions"


@pytest.mark.parametrize("paper", sorted(OLD_FORM))
def test_an_old_form_paper_is_still_in_the_old_form(paper):
    """A rewritten paper leaves OLD_FORM, so its new form is checked (#4202)."""
    assert (PAPERS / paper / "results.jsonl").exists(), f"{paper}: not a paper"
    assert "Related work" not in _headings(paper), f"{paper}: rewritten, remove from OLD_FORM"


@pytest.mark.parametrize("paper", NEW_FORM)
def test_figures_and_tables_are_numbered_and_cited(paper):
    text = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    body = text.split("\n## References\n", 1)[0]
    for kind, count in (("Figure", body.count("![")), ("Table", body.count("\n| --- "))):
        for n in range(1, count + 1):
            assert f"**{kind} {n}.**" in body, f"{paper}: caption {kind} {n}"
            assert body.count(f"{kind} {n}") >= 2, f"{paper}: cite {kind} {n} in the text"


@pytest.mark.parametrize("paper", sorted(OLD_FORM))
def test_paper_says_why_it_was_written_and_its_question(paper):
    """Below its opening paragraph, a paper says why it was written and what it asks (#3138)."""
    text = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    top = text.split("\n## ", 1)[0]
    # The status line under the abstract (#3150) doesn't count as a paragraph here.
    paragraphs = [p for p in top.split("\n\n") if p.strip() and not p.startswith("**Status:**")]
    why = [p for p in paragraphs if p.startswith("**Why we wrote it.**")]
    assert why, f"{paper}: no '**Why we wrote it.**' paragraph before the first section"
    assert "**The question.**" in why[0], f"{paper}: the why paragraph names no question"
    assert paragraphs.index(why[0]) == 2, f"{paper}: put it right after the opening paragraph"
