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
# Every written paper is in the new form (#4202): the last old-form paper was rewritten in #4250.
NEW_FORM = WRITTEN


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


@pytest.mark.parametrize("paper", NEW_FORM)
def test_figures_and_tables_are_numbered_and_cited(paper):
    text = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    body = text.split("\n## References\n", 1)[0]
    for kind, count in (("Figure", body.count("![")), ("Table", body.count("\n| --- "))):
        for n in range(1, count + 1):
            assert f"**{kind} {n}.**" in body, f"{paper}: caption {kind} {n}"
            assert body.count(f"{kind} {n}") >= 2, f"{paper}: cite {kind} {n} in the text"


# Maths written as Unicode text, outside code and LaTeX (write-paper skill, Mathematics, #4211).
# A ratio such as "1.9×" or "3–22×" stays text, so a "×" right after a number passes.
UNICODE_MATHS = re.compile(
    r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻₀₁₂₃₄₅₆₇₈₉√∑∏∫≤≥≈≠±∈αβγδεθλμπρστφχψω]|(?<![\d)])\s*×|×(?=\s*[\d(])"
)


def _prose(paper: str) -> str:
    """The paper's text without code blocks, inline code and LaTeX."""
    text = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    text = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    return re.sub(r"\$[^$\n]+\$", "", text)


@pytest.mark.parametrize("paper", sorted(p.parent.name for p in PAPERS.glob("*/README.md")))
def test_maths_is_written_in_latex(paper):
    found = [line.strip()[:80] for line in _prose(paper).splitlines() if UNICODE_MATHS.search(line)]
    assert not found, f"write these in LaTeX ($...$): {found}"


def _cited_works():
    import importlib.util
    import sys

    path = PAPERS.parent / "scripts" / "cited_works.py"
    spec = importlib.util.spec_from_file_location("cited_works", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cited_works"] = module
    spec.loader.exec_module(module)
    return module


def test_every_cited_work_has_a_review_page_or_waits_for_one():
    # #5346: a cited paper needs a page in docs/literature/, or a line in the awaiting list.
    assert _cited_works().problems() == []


def test_a_new_uncovered_reference_and_a_stale_line_both_fail(tmp_path):
    works = _cited_works()
    papers, literature = tmp_path / "papers", tmp_path / "literature"
    (papers / "p").mkdir(parents=True)
    literature.mkdir()
    (papers / "p" / "README.md").write_text(
        "# P\n\n## References\n\n"
        "- A, B. (2020). *Covered*. [arXiv:2001.00001](https://arxiv.org/abs/2001.00001).\n"
        "- C, D. (2021). *Waiting*. [arXiv:2101.00002](https://arxiv.org/abs/2101.00002).\n"
        "- E, F. (2022). *New*. [doi](https://doi.org/10.1/abc).\n"
    )
    (literature / "covered.md").write_text("# Covered\n\n**Citation.** [arXiv:2001.00001](x).\n")
    awaiting = tmp_path / "awaiting.txt"
    awaiting.write_text("# comment\n2101.00002 #1\n2001.00001 #2\n9999.99999 #3\n")
    found = works.problems(papers, literature, awaiting)
    assert found == [
        "p: 10.1/abc has no page in docs/literature/ and no line in awaiting.txt",
        "awaiting.txt: 2001.00001 has a page now; delete its line",
        "awaiting.txt: 9999.99999 is cited by no paper; delete its line",
    ]
