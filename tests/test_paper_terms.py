"""A paper uses plain words, not the project's own terms (scripts/paper_terms.py, #6851)."""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
terms = importlib.import_module("paper_terms")

TERMS = """
[[term]]
name = "widget"
pattern = '(?i)(?<!spare )\\bwidgets?\\b'
plain = "a part"
"""
PAPER = """# A paper about widgets

The abstract names a widget.

**Status:** Preprint. Work left: #1 (more).

## Method

We read a spare widget, and `code with a widget`. See [the parts](widgets/README.md).

```bash
make widget
```

## References

- Someone (2020). *Widgets*.

## Appendices

### Reproduce

make widgets
"""


def project(tmp_path: Path, paper: str = PAPER, pending: str = "") -> Path:
    """A papers folder with one listed preprint, its figure and its thread."""
    papers = tmp_path / "papers"
    (papers / "a" / "media").mkdir(parents=True, exist_ok=True)
    (papers / terms.TERMS).write_text(TERMS)
    (papers / "a" / "README.md").write_text(paper)
    svg = (
        '<svg><text class="title">A Widget&#x27;s rise</text><text class="axis">widget</text></svg>'
    )
    (papers / "a" / "media" / "rise.svg").write_text(svg)
    (papers / "a" / "thread.yaml").write_text("- text: one widget\n")
    (papers / terms.PENDING).write_text(pending)
    return papers


def test_the_check_reads_prose_figure_titles_and_the_thread_but_not_code_or_references(tmp_path):
    found = terms.found(project(tmp_path) / "a", terms.load_terms(project(tmp_path)))
    assert found == [
        "README.md: 'widgets' (use a part)",  # the title
        "README.md: 'widget' (use a part)",  # the abstract
        "README.md: 'widget' (use a part)",  # inline code counts: a reader sees it
        "media/rise.svg: 'Widget' (use a part)",  # a figure's title, not its axis
        "thread.yaml: 'widget' (use a part)",
    ]  # not "spare widget", the link target, the code block, References or Reproduce


def test_a_preprint_fails_until_it_is_pending_and_a_clean_one_leaves_the_list(tmp_path):
    papers = project(tmp_path)
    assert len(terms.problems(papers, ["a"])) == 5
    papers = project(tmp_path, pending="# waits\na\n")
    assert terms.problems(papers, ["a"]) == []  # it waits for its rewrite
    clean = PAPER.replace("widgets", "parts").replace("widget", "part")
    papers = project(tmp_path, clean, pending="a\n")
    (papers / "a" / "media" / "rise.svg").write_text('<svg><text class="title">Rise</text></svg>')
    (papers / "a" / "thread.yaml").write_text("- text: one part\n")
    assert terms.problems(papers, ["a"]) == [
        f"a: no project term is left, so delete its line from {terms.PENDING}"
    ]
    assert terms.problems(papers, []) == [f"a: in {terms.PENDING}, but not a listed preprint"]


def test_a_published_paper_is_frozen_and_not_checked(tmp_path):
    published = PAPER.replace(
        "**Status:** Preprint. Work left: #1 (more).", "**Status:** Published 2026-11-01."
    )
    assert terms.problems(project(tmp_path, published), ["a"]) == []


def test_a_project_with_no_term_list_has_no_terms(tmp_path):
    assert terms.load_terms(tmp_path) == []
    assert terms.pending(tmp_path) == set()


def test_this_repositorys_preprints_are_clean_or_pending():
    papers = ROOT / "papers"
    assert terms.load_terms(papers), "papers/terms.toml lists the project's terms"
    assert terms.problems(papers, [slug for _, slug in _entries(papers)]) == []


def _entries(papers: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("papers").entries(papers)
