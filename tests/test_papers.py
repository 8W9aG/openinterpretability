"""The references of the papers in papers/."""

import re
from pathlib import Path

import pytest

PAPERS = Path(__file__).parents[1] / "papers"
WRITTEN = sorted(path.parent.name for path in PAPERS.glob("*/results.jsonl"))
CITATION = re.compile(
    r"((?:[A-Z][\w-]+ )*[A-Z][\w-]+)(?: et al\.| & [A-Z][\w ]*?[a-z])?(?:, | \()((?:19|20)\d{2})\b"
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
