"""Atom feeds of the papers and the literature reviews (labinacan/site/research_feeds.py, #4256)."""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from labinacan.site.literature_pages import LITERATURE, load_reviews
from labinacan.site.research import load_papers
from labinacan.site.research_feeds import (
    LITERATURE_FEED,
    NO_DATE,
    PAPERS_FEED,
    PUBLISHED_FEED,
    added_times,
    literature_feed,
    papers_feed,
    write_research_feeds,
)
from openinterpretability.builder.compiler import compile_site

ATOM = "{http://www.w3.org/2005/Atom}"
SITE = "https://example.org/"


def _entries(text: str) -> list[dict]:
    root = ET.fromstring(text)
    assert root.tag == f"{ATOM}feed" and root.find(f"{ATOM}id") is not None
    return [
        {
            "title": e.find(f"{ATOM}title").text,
            "href": e.find(f"{ATOM}link").get("href"),
            "updated": e.find(f"{ATOM}updated").text,
            "summary": e.find(f"{ATOM}summary").text,
        }
        for e in root.findall(f"{ATOM}entry")
    ]


def test_the_papers_feed_lists_every_paper_newest_first():
    papers = load_papers(Path("papers"), "https://example.org/repo")
    entries = _entries(papers_feed(papers, SITE, "OI"))
    assert len(entries) == len(papers)
    dates = [e["updated"] for e in entries]
    assert dates == sorted(dates, reverse=True) and NO_DATE not in dates
    assert entries[0]["href"] == f"{SITE}research/{papers[0].href}"
    assert entries[0]["summary"] == papers[0].summary  # the abstract


def test_the_literature_feed_lists_every_review_with_its_status():
    reviews = load_reviews("https://example.org/repo")
    entries = _entries(literature_feed(reviews, LITERATURE, SITE, "OI"))
    names = {r.name for r in reviews} - {"README", "queue"}
    assert len(entries) == len(names)
    assert all(e["href"].startswith(f"{SITE}research/literature/") for e in entries)
    assert sum(e["summary"].split(":")[0] in ("Used", "Planned", "Not applicable")
               for e in entries if e["summary"]) > len(entries) / 2  # fmt: skip
    dates = [e["updated"] for e in entries]
    assert dates == sorted(dates, reverse=True)


def _git(path, *args, when):
    env = {"GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when, "PATH": os.environ["PATH"]}
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, env=env)


def test_a_review_is_dated_by_the_commit_that_added_it(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main", when="2026-11-01T00:00:00Z")
    _git(tmp_path, "config", "user.email", "t@example.com", when="2026-11-01T00:00:00Z")
    _git(tmp_path, "config", "user.name", "t", when="2026-11-01T00:00:00Z")
    (tmp_path / "a.md").write_text("# A\n")
    _git(tmp_path, "add", "a.md", when="2026-11-01T00:00:00Z")
    _git(tmp_path, "commit", "-q", "-m", "a", when="2026-11-01T00:00:00Z")
    (tmp_path / "a.md").write_text("# A, edited\n")  # an edit doesn't move the date
    (tmp_path / "b.md").write_text("# B\n")
    _git(tmp_path, "add", "a.md", "b.md", when="2026-11-03T12:00:00Z")
    _git(tmp_path, "commit", "-q", "-m", "b", when="2026-11-03T12:00:00Z")
    assert added_times(tmp_path) == {"a": "2026-11-01T00:00:00Z", "b": "2026-11-03T12:00:00Z"}
    assert added_times(tmp_path / "missing") == {}  # no repository: no dates


def test_the_build_writes_both_feeds_identically_and_their_links_resolve(tmp_path, record):
    from tests.test_compiler import experiment

    records = [experiment(record, idx=1, status="VERIFIED", counts=(10, 0, 5, 0))]
    out = []
    for name in ("one", "two"):
        compile_site(records, tmp_path / name, papers_dir=Path("papers"))
        out.append(tmp_path / name)
    for feed in ("research/papers.xml", "research/literature.xml"):
        first, second = ((d / feed).read_bytes() for d in out)
        assert first == second  # deterministic
        for entry in _entries(first.decode()):
            page = out[0] / entry["href"].split("://", 1)[1].split("/", 1)[1]
            assert page.exists(), entry["href"]
    research = (out[0] / "research" / "index.html").read_text()
    assert "papers.xml" in research and "literature.xml" in research


def test_with_no_reviews_the_build_writes_no_literature_feed(tmp_path):
    common = {"site_url": SITE, "site_title": "OI"}
    write_research_feeds(tmp_path, [], [], LITERATURE, common)
    assert _entries((tmp_path / PAPERS_FEED).read_text(encoding="utf-8")) == []
    assert _entries((tmp_path / PUBLISHED_FEED).read_text(encoding="utf-8")) == []
    assert not (tmp_path / LITERATURE_FEED).exists()


def test_the_posts_feed_links_each_posts_page_figure_and_thread():
    from types import SimpleNamespace

    from labinacan.site.research_feeds import posts_feed
    from labinacan.social.thread import Prepared

    tweet = Prepared("See {section_link}", "m", (("social-c.png", "image/png", "A"),))
    post = SimpleNamespace(published="2026-11-20", title="Hello", href="hello/index.html",
                           summary="Hi.", social_image="media/social.png", social_video=None,
                           hero_alt="A curve", thread=(tweet,))  # fmt: skip
    xml = posts_feed([post], "https://example.org/", "Site")
    assert "https://example.org/posts/hello/media/social.png" in xml  # the head figure
    assert "https://example.org/posts/hello/index.html#m" in xml  # the thread's link
    assert "https://example.org/posts/hello/media/social-c.png" in xml  # the thread's figure
    assert 'href="https://example.org/posts/index.html"' in xml  # the section's page


def test_an_unpublished_paper_or_post_takes_the_no_date_date():
    from types import SimpleNamespace

    from labinacan.site.research_feeds import NO_DATE, papers_feed, posts_feed

    blank = dict(social_image=None, social_video=None, hero_alt="", thread=())
    item = SimpleNamespace(published=None, title="T", href="t/index.html", summary="S.", **blank)
    for xml in (
        papers_feed([item], "https://x.org/", "X"),
        posts_feed([item], "https://x.org/", "X"),
    ):
        assert f"<updated>{NO_DATE}</updated>" in xml


def test_the_literature_feed_dates_and_sums_up_each_review(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from labinacan.site import research_feeds as rf

    read = []
    monkeypatch.setattr(rf, "review_rows", lambda folder: {"a": ("used", "it fits")})
    times = {"a": "2026-11-01T00:00:00Z"}
    monkeypatch.setattr(rf, "added_times", lambda folder: read.append(folder) or times)
    reviews = [SimpleNamespace(name=n, title=n.upper(), href=f"l/{n}.html") for n in "ab"]
    xml = rf.literature_feed(reviews, tmp_path, "https://x.org/", "X")
    assert read == [tmp_path]
    a, b = xml.split("<entry>")[1:]
    assert (
        "<updated>2026-11-01T00:00:00Z</updated>" in a and "<summary>Used: it fits</summary>" in a
    )
    assert f"<updated>{rf.NO_DATE}</updated>" in b and "<summary></summary>" in b
