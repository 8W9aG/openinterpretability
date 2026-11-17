"""Papers are preprints or published, each with a citation file (scripts/papers.py, #3150)."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import papers  # noqa: E402

PAPER = "# A title\n\nThe abstract.\n\n## Results\n\nOne.\n"


def _papers(tmp_path):
    root = tmp_path / "papers"
    (root / "p").mkdir(parents=True)
    (root / "README.md").write_text("- 2026-11-01: [A title](p/README.md)\n")
    (root / "p" / "README.md").write_text(PAPER)
    (root / "p" / "results.jsonl").write_text('{"x": 1}\n')
    return root


def test_the_repo_papers_pass():
    assert papers.check() == []


def test_a_paper_needs_a_status_and_a_citation(tmp_path):
    root = _papers(tmp_path)
    assert papers.check(root) == ["p: no status line under the abstract "
                                  "(**Status:** Preprint. Work left: #N …)"]  # fmt: skip
    papers.new("p", "#12 (more seeds).", root)
    text = (root / "p" / "README.md").read_text()
    assert text.startswith("# A title\n\nThe abstract.\n\n**Status:** Preprint. Work left: #12")
    assert papers.check(root) == []
    (root / "p" / "README.md").write_text(text.replace("# A title", "# Another title"))
    assert any("title" in p for p in papers.check(root))


def test_a_published_paper_changes_only_by_addenda(tmp_path):
    root = _papers(tmp_path)
    papers.new("p", "#12 (more seeds).", root)
    papers.publish("p", root, today="2026-11-20")
    readme = root / "p" / "README.md"
    assert "**Status:** Published 2026-11-20." in readme.read_text()
    assert papers.check(root) == []
    readme.write_text(readme.read_text() + "\n## Addenda\n\n### 2026-12-01\n\nA fix.\n")
    assert papers.check(root) == []  # an addendum is allowed
    readme.write_text(readme.read_text().replace("One.", "Two."))
    (problem,) = papers.check(root)
    assert problem == "p: the published text or results changed: add an addendum instead"


def test_a_changed_result_of_a_published_paper_fails(tmp_path):
    root = _papers(tmp_path)
    papers.new("p", "#12 (x).", root)
    papers.publish("p", root, today="2026-11-20")
    (root / "p" / "results.jsonl").write_text('{"x": 2}\n')
    assert len(papers.check(root)) == 1


def test_the_site_shows_the_status(tmp_path):
    from openinterpretability.builder.compiler import compile_site

    root = _papers(tmp_path)
    papers.new("p", "#12 (more seeds, see #3).", root)
    compile_site([], tmp_path / "site", papers_dir=root)
    page = (tmp_path / "site" / "research" / "p" / "index.html").read_text()
    assert "Preprint</span> work left:" in page and "/issues/12" in page
    assert '/issues/3"' not in page  # a reference in brackets is not work left
    assert "papers/p/CITATION.cff" in page
    assert "Preprint" in (tmp_path / "site" / "research" / "index.html").read_text()
