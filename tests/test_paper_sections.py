"""Section ids and the contents list of a paper page (builder/research.py, #3139)."""

from openinterpretability.builder.contents import anchor as _anchor
from openinterpretability.builder.research import _read_paper

PAPER = """# A title

The abstract.

## Introduction

Text.

## Does it hold?

More.

## Results

One.

## Results

Two.

## Next

Three.
"""


def _paper(tmp_path, text=PAPER, interactive=False):
    folder = tmp_path / "p"
    folder.mkdir()
    (folder / "README.md").write_text(text, encoding="utf-8")
    if interactive:
        (folder / "index.html").write_text("<html></html>")
    return _read_paper(folder, "https://example.com/repo")


def test_each_section_gets_a_stable_unique_id(tmp_path):
    paper = _paper(tmp_path)
    assert paper.sections == (
        ("introduction", "Introduction"),
        ("does-it-hold", "Does it hold?"),
        ("results", "Results"),
        ("results-2", "Results"),
        ("next", "Next"),
    )
    for anchor, _ in paper.sections:
        assert f'<h2 id="{anchor}">' in paper.body


def test_an_anchor_has_only_safe_characters():
    assert _anchor("Café & <b>bold</b>!", set()) == "caf-b-bold-b"
    assert _anchor("***", set()) == "section"


def test_an_interactive_paper_has_no_sections(tmp_path):
    assert _paper(tmp_path, interactive=True).sections == ()


def test_the_built_page_links_every_section(tmp_path):
    from openinterpretability.builder import compiler

    papers = tmp_path / "papers"
    (papers / "p").mkdir(parents=True)
    (papers / "p" / "README.md").write_text(PAPER, encoding="utf-8")
    (papers / "README.md").write_text("- [A title](p/README.md)\n", encoding="utf-8")
    out = tmp_path / "site"
    compiler.compile_site([], out, papers_dir=papers)
    page = (out / "research" / "p" / "index.html").read_text(encoding="utf-8")
    assert 'aria-label="Contents"' in page
    for anchor in ("introduction", "does-it-hold", "results", "results-2"):
        assert f'href="#{anchor}"' in page and f'id="{anchor}"' in page
