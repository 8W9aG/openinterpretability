"""The site's Posts section: its index, each post, their media and their feed (#4650)."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from click.testing import CliRunner

from openinterpretability.builder.compiler import compile_site
from openinterpretability.builder.research import load_papers

ATOM = "{http://www.w3.org/2005/Atom}"
POSTS = Path(__file__).parents[1] / "posts"


def _posts(root: Path) -> Path:
    posts = root / "posts"
    for slug, title in (("old", "Old post"), ("new", "New post")):
        (posts / slug / "media").mkdir(parents=True)
        (posts / slug / "media" / "fig.svg").write_text("<svg/>")
        (posts / slug / "README.md").write_text(
            f"# {title}\n\nThe summary of {slug}.\n\n![A figure](media/fig.svg)\n\n"
            "See [the other](../old/README.md) and [code](../../scripts/x.py).\n"
        )
    (posts / "README.md").write_text(
        "# Posts\n\n- 2026-11-20: [New post](new/README.md)\n"
        "- 2026-11-01: [Old post](old/README.md)\n"
    )
    return posts


def test_the_build_writes_the_index_each_post_its_media_and_the_feed(tmp_path):
    out = tmp_path / "dist"
    compile_site([], out, posts_dir=_posts(tmp_path))

    index = (out / "posts" / "index.html").read_text()
    assert index.index("New post") < index.index("Old post")
    assert "The summary of new." in index and 'datetime="2026-11-20"' in index
    page = (out / "posts" / "new" / "index.html").read_text()
    assert 'href="../old/index.html"' in page
    assert "/blob/main/posts/new/../../scripts/x.py" not in page
    assert "/blob/main/scripts/x.py" in page
    assert (out / "posts" / "new" / "media" / "fig.svg").exists()
    assert 'href="posts/index.html"' in (out / "index.html").read_text()  # the navigation

    feed = ET.fromstring((out / "posts" / "posts.xml").read_text())
    entries = feed.findall(f"{ATOM}entry")
    assert [e.find(f"{ATOM}title").text for e in entries] == ["New post", "Old post"]
    assert entries[0].find(f"{ATOM}updated").text == "2026-11-20T00:00:00Z"
    assert entries[0].find(f"{ATOM}link").get("href").endswith("posts/new/index.html")


def test_the_section_is_empty_without_posts(tmp_path):
    compile_site([], tmp_path)
    assert "No posts yet." in (tmp_path / "posts" / "index.html").read_text()
    assert "<entry>" not in (tmp_path / "posts" / "posts.xml").read_text()


def test_the_repository_posts_are_dated_and_newest_first():
    posts = load_papers(POSTS, "https://example.org/repo", source="posts")
    dates = [p.published for p in posts]
    assert posts and all(dates) and dates == sorted(dates, reverse=True)


def test_every_flag_the_explain_post_names_is_an_explain_option():
    from openinterpretability.cli import cli

    text = (POSTS / "explain" / "README.md").read_text(encoding="utf-8")
    commands = re.findall(r"open-interpretability explain[^\n`]*", text)
    named = {flag for line in commands for flag in re.findall(r"--[a-z][a-z-]*", line)}
    assert named, "the post names no flag"
    shown = CliRunner().invoke(cli, ["explain", "--help"], terminal_width=200)
    assert shown.exit_code == 0
    options = set(re.findall(r"--[a-z][a-z-]*", shown.output))
    assert named <= options, f"not options of explain: {sorted(named - options)}"
