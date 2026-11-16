"""The references of the papers in papers/."""

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
        assert any(name.endswith(n) and year == y for n, y in listed), f"{name}, {year}"


@pytest.mark.parametrize("paper", WRITTEN)
def test_each_reference_is_cited(paper):
    body, references = _sections(paper)
    cited = CITATION.findall(body)
    for name, year, _ in REFERENCE.findall(references):
        assert any(c.endswith(name) and y == year for c, y in cited), f"{name}, {year}"


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


@pytest.mark.parametrize("paper", WRITTEN)
def test_paper_says_why_it_was_written_and_its_question(paper):
    """Below its opening paragraph, a paper says why it was written and what it asks (#3138)."""
    text = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    top = text.split("\n## ", 1)[0]
    paragraphs = [p for p in top.split("\n\n") if p.strip()]
    why = [p for p in paragraphs if p.startswith("**Why we wrote it.**")]
    assert why, f"{paper}: no '**Why we wrote it.**' paragraph before the first section"
    assert "**The question.**" in why[0], f"{paper}: the why paragraph names no question"
    assert paragraphs.index(why[0]) == 2, f"{paper}: put it right after the opening paragraph"
