"""The literature review as a subsection of Research, one page per review (#3293)."""

import json

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
    from openinterpretability.builder.literature_map_view import literature_map
    from openinterpretability.builder.literature_pages import statuses

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


def test_callouts_and_badges_never_overlap():
    """Cards stack in the margins; badges overlap no badge and no card (#4577)."""
    from openinterpretability.builder.cluster_map import overlaps
    from openinterpretability.builder.literature_map_view import Point, badges, callouts

    title = "A long title for one paper that wraps onto more than one line"
    near = [Point(n, title, "used", 300 + i, 300 + i, credit="A, B and C, 2026")
            for i, n in enumerate("abcd")]  # fmt: skip
    cards = callouts(near)
    boxes = [c.box for c in cards]
    assert not any(overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    regions = [{"number": n, "name": f"r{n}", "x": 0.1, "y": 0.5, "size": 3}
               for n in (1, 2, 3)]  # fmt: skip
    placed = badges(regions, cards)
    every = [b.box for b in placed]
    assert not any(overlaps(a, b) for i, a in enumerate(every) for b in every[i + 1 :])
    assert not any(overlaps(a, c) for a in every for c in boxes)


def test_the_repo_map_covers_every_review():
    # Run scripts/literature_map.py after adding or changing a review (#3294).
    from openinterpretability.builder.cluster_map import overlaps
    from openinterpretability.builder.literature_map_view import literature_map, map_view
    from openinterpretability.builder.literature_pages import LITERATURE

    points, missing = literature_map(LITERATURE, load_reviews(REPO, LITERATURE))
    assert missing == [], "run scripts/literature_map.py: " + ", ".join(r.name for r in missing)
    assert len(points) >= 20
    view = map_view(LITERATURE, points)
    boxes = [b.box for b in view["badges"]] + [c.box for c in view["callouts"]]
    assert not any(overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])


def test_the_repo_map_has_regions_neighbours_and_the_queue():
    """map.json's fields since #4577: named regions, each review's neighbours, the queue."""
    from openinterpretability.builder.literature_pages import LITERATURE, MAP, sections

    data = json.loads((LITERATURE / MAP).read_text())
    regions = data["regions"]
    assert [r["number"] for r in regions] == list(range(1, len(regions) + 1))
    assert {"number", "name", "x", "y", "size", "touches"} <= set(regions[0])
    names = [r["name"] for r in regions]
    parts = {part for name in names for part in name.split(" · ")}  # a tie names both (#5798)
    assert len(names) == len(set(names)) and parts <= set(sections(LITERATURE).values())
    numbers = {r["number"] for r in regions}
    assert all(r["region"] in numbers and len(r["neighbours"]) == 3 for r in data["reviews"])
    assert data["queued"] and all({"number", "title", "x", "y", "region"} <= set(q)
                                  for q in data["queued"])  # fmt: skip


def test_index_rows_credits_and_queued_titles_are_redacted(tmp_path):
    from openinterpretability.builder.literature_map_view import map_view
    from openinterpretability.builder.literature_pages import credits, review_rows
    from openinterpretability.builder.pii import REMOVED_TEXT

    mail = "jane.doe@somemail.org"  # an address the site's redaction removes (test_pii.py)
    folder = tmp_path / "literature"
    folder.mkdir()
    (folder / "README.md").write_text(
        "| Page | Status | Why |\n| --- | --- | --- |\n"
        f"| [A paper](a-paper.md) (Doe, {mail}, 2026) | planned | Write to {mail}. |\n"
    )
    queued = [{"number": 1, "title": f"Review by {mail}", "x": 0.5, "y": 0.5}]
    (folder / "map.json").write_text(json.dumps({"queued": queued, "regions": []}))
    status, why = review_rows(folder)["a-paper"]
    assert status == "planned" and mail not in why and REMOVED_TEXT in why
    assert mail not in credits(folder)["a-paper"]
    title = map_view(folder, [])["queued"][0].title
    assert mail not in title and REMOVED_TEXT in title


def test_a_review_without_a_level_1_title_takes_its_file_name(tmp_path):
    folder = tmp_path / "literature"
    folder.mkdir()
    (folder / "README.md").write_text("# Literature review\n")
    (folder / "untitled.md").write_text("## Notes\n\nNo top heading.\n")
    _, review = load_reviews(REPO, folder)
    assert review.title == "untitled"
    assert "<h2>Notes</h2>" in review.body
