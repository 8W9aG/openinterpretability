"""The Research section lists papers newest first, by the date in papers/README.md (#3292)."""

from pathlib import Path

from openinterpretability.builder.research import load_papers


def _paper(root: Path, slug: str) -> None:
    (root / slug).mkdir()
    (root / slug / "README.md").write_text(f"# {slug}\n\nThe abstract of {slug}.\n")


def test_papers_go_newest_first_ties_keep_the_list_order_and_undated_go_last(tmp_path):
    for slug in ("old", "tie-a", "tie-b", "new", "undated"):
        _paper(tmp_path, slug)
    (tmp_path / "README.md").write_text(
        "# Papers\n\n"
        "- 2026-11-01: [Old](old/README.md)\n"
        "- [Undated](undated/README.md)\n"
        "- 2026-11-10: [Tie A](tie-a/README.md)\n"
        "- 2026-11-10: [Tie B](tie-b/README.md), with a note\n"
        "- 2026-11-16: [New](new/README.md)\n"
    )
    papers = load_papers(tmp_path, "https://example.org/repo")
    assert [p.slug for p in papers] == ["new", "tie-a", "tie-b", "old", "undated"]
    assert [p.published for p in papers][:2] == ["2026-11-16", "2026-11-10"]
    assert papers[-1].published is None


def test_the_repository_index_is_dated_and_newest_first():
    papers = load_papers(Path("papers"), "https://example.org/repo")
    dates = [p.published for p in papers]
    assert all(dates) and dates == sorted(dates, reverse=True)
