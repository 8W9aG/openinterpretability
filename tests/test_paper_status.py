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


def test_a_preprint_abstract_over_the_cap_fails_and_a_published_one_is_left(tmp_path):
    root = _papers(tmp_path)
    readme = root / "p" / "README.md"
    long = " ".join(["word"] * (papers.ABSTRACT_WORDS - 1) + ["$a + b = c$"])
    readme.write_text(PAPER.replace("The abstract.", long))
    papers.new("p", "#12 (x).", root)
    assert papers.check(root) == []  # at the cap, with the formula as one word
    readme.write_text(readme.read_text().replace(long, long + " more"))
    assert papers.check(root) == [
        f"p: the abstract has {papers.ABSTRACT_WORDS + 1} words, "
        f"over the cap of {papers.ABSTRACT_WORDS} (write-paper)"
    ]
    papers.publish("p", root, today="2026-11-20")
    assert papers.check(root) == []  # a published paper is frozen, whatever its length


def _figure(tmp_path, caption, title_lines=("A rises,", "and B falls")):
    """A preprint with one figure, whose SVG title has ``title_lines``."""
    root = _papers(tmp_path)
    media = root / "p" / "media"
    media.mkdir()
    texts = "".join(f'<text x="0" class="title">{t}</text>' for t in title_lines)
    (media / "f.svg").write_text(f"<svg>{texts}<text class='sub'>marks</text></svg>")
    readme = root / "p" / "README.md"
    readme.write_text(PAPER + f"\n![A rises](media/f.svg)\n\n**Figure 1.** {caption}\n")
    papers.new("p", "#12 (x).", root)
    return root


def test_a_preprint_caption_opens_with_its_figures_finding(tmp_path):
    root = _figure(tmp_path, "A rises, and B falls. Each mark is one run.")
    assert papers.check(root) == []


def test_a_caption_that_only_describes_the_marks_fails(tmp_path):
    root = _figure(tmp_path, "Each mark is one run.")
    assert papers.check(root) == [
        "p: Figure 1's caption does not open with its finding: 'A rises, and B falls.'"
    ]


def test_a_title_with_a_question_mark_keeps_it(tmp_path):
    root = _figure(tmp_path, "Does A rise? It does.", ("Does A rise?",))
    assert papers.check(root) == []


def test_a_published_papers_captions_are_left_as_they_are(tmp_path):
    root = _figure(tmp_path, "Each mark is one run.")
    papers.publish("p", root, today="2026-11-20")
    assert papers.check(root) == []
