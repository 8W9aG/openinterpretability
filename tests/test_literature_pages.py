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


def _mapped(tmp_path, sha=None):
    """A folder with two reviews in the index table, one of them in map.json."""
    import hashlib
    import json

    from openinterpretability.builder.literature_pages import MAP, map_text

    folder = tmp_path / "literature"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(
        "# Literature review\n\n| Page | Status | Why |\n| --- | --- | --- |\n"
        "| [A](a.md) (X, 2026) | used | Why. |\n| [B](b.md) (Y, 2025) | planned | Why. |\n"
    )
    for name in "ab":
        (folder / f"{name}.md").write_text(f"# Paper {name}\n\n## Summary\n\nAbout {name}.\n")
    digest = sha or hashlib.sha256(map_text(folder / "a.md").encode()).hexdigest()
    row = {"name": "a", "x": 0.5, "y": 0.5, "sha256": digest}
    (folder / MAP).write_text(json.dumps({"model": "m", "reviews": [row]}))
    return folder


def test_the_map_places_reviews_whose_text_matches_and_lists_the_rest(tmp_path):
    from openinterpretability.builder.literature_pages import literature_map, statuses

    folder = _mapped(tmp_path)
    assert statuses(folder) == {"a": "used", "b": "planned"}
    points, missing = literature_map(folder, load_reviews(REPO, folder))
    assert [(p.name, p.status) for p in points] == [("a", "used")]
    assert [r.name for r in missing] == ["b"]  # not in map.json; the index has no Summary
    changed = _mapped(tmp_path / "changed", sha="0" * 64)  # a's text changed since the map
    points, missing = literature_map(changed, load_reviews(REPO, changed))
    assert points == [] and [r.name for r in missing] == ["a", "b"]


def test_the_index_draws_the_map_and_the_reviews_not_placed(tmp_path):
    from openinterpretability.builder.literature_pages import render_reviews

    folder, pages = _mapped(tmp_path), {}
    render_reviews(lambda href, template, **kw: pages.update({href: kw}),
                   load_reviews(REPO, folder), folder)  # fmt: skip
    index = pages["research/literature/index.html"]
    assert [p.name for p in index["points"]] == ["a"]
    assert [r.name for r in index["missing"]] == ["b"]
    assert "points" not in pages["research/literature/a.html"]


def test_labels_do_not_overlap_when_a_free_spot_exists():
    from openinterpretability.builder.literature_pages import Point, _box, _overlaps, place_labels

    title = "A long title for one paper"
    near = [Point(n, title, "used", 100, 100 + i) for i, n in enumerate("abc")]
    placed = place_labels(near)
    boxes = [_box(p, p.tx, p.ty, p.anchor) for p in placed]
    assert not any(_overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    assert sum(p.moved for p in placed) >= 1 and not placed[0].moved
    assert Point("x", "y" * 50, "used", 0, 0).label.endswith("…")


def test_the_repo_map_covers_every_review():
    # Run scripts/literature_map.py after adding or changing a review (#3294).
    from openinterpretability.builder.literature_pages import LITERATURE, literature_map

    points, missing = literature_map(LITERATURE, load_reviews(REPO, LITERATURE))
    assert missing == [], "run scripts/literature_map.py: " + ", ".join(r.name for r in missing)
    assert len(points) >= 20
