"""The literature review as a subsection of Research, one page per review (#3293)."""

from openinterpretability.builder.compiler import compile_site
from openinterpretability.builder.literature_pages import load_reviews

REPO = "https://github.com/o/r"


def _folder(tmp_path):
    folder = tmp_path / "literature"
    folder.mkdir()
    (folder / "README.md").write_text(
        "# Literature review\n\nSee [A paper](a-paper.md), the [queue](queue.md#top), "
        "[the skill](../../.claude/skills/x/SKILL.md) and [the web](https://example.org).\n"
    )
    (folder / "a-paper.md").write_text("# A Paper (2026)\n\nBack to [the index](README.md).\n")
    (folder / "queue.md").write_text("# Queue\n\nNothing yet.\n")
    return folder


def test_each_file_is_a_page_and_links_between_them_stay_on_the_site(tmp_path):
    index, paper, queue = load_reviews(REPO, _folder(tmp_path))
    assert [r.href for r in (index, paper, queue)] == [
        "research/literature/index.html",
        "research/literature/a-paper.html",
        "research/literature/queue.html",
    ]
    assert (index.title, paper.title) == ("Literature review", "A Paper (2026)")
    assert 'href="a-paper.html"' in index.body and 'href="queue.html#top"' in index.body
    assert f'href="{REPO}/blob/main/.claude/skills/x/SKILL.md"' in index.body
    assert 'href="https://example.org"' in index.body
    assert 'href="index.html"' in paper.body and "<h1>" not in paper.body
    assert load_reviews(REPO, tmp_path / "missing") == []


def test_the_site_has_the_review_under_research(tmp_path, record):
    compile_site([record], tmp_path / "site")
    section = tmp_path / "site" / "research" / "literature"
    pages = {p.name for p in section.glob("*.html")}
    assert "index.html" in pages and "queue.html" in pages and len(pages) > 10
    research = (tmp_path / "site" / "research" / "index.html").read_text()
    assert 'href="literature/index.html"' in research
    page = (section / "beyond-top-activations.html").read_text()
    assert "Literature review</a>" in page and "docs/literature/beyond-top-activations.md" in page
