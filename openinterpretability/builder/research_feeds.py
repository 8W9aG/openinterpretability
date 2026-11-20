"""Atom feeds of the Research section: one of the papers, one of the literature reviews (#4256).

``feed.xml`` (:mod:`openinterpretability.builder.feeds`) lists new VERIFIED units. These two
feeds let a reader follow the papers and the reviews apart from it. As there, each date comes
from the repository, never the clock, so an unchanged repository rebuilds them byte for byte:

- a paper's date is the one before its entry in ``papers/README.md``;
- a review's date is the time of the commit that added its file. The deploy checks out the
  full history. Without it (an install, a shallow clone), a review gets :data:`NO_DATE`.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

from openinterpretability.builder.literature_pages import review_rows

PAPERS_FEED = "research/papers.xml"
LITERATURE_FEED = "research/literature.xml"
PUBLISHED_FEED = "research/published.xml"  # published papers only, for posts (#4463)
POSTS_FEED = "posts/posts.xml"  # the Posts section (#4650)
NO_DATE = "1970-01-01T00:00:00Z"  # a date the repository can't give
NOT_REVIEWS = ("README", "queue")  # pages of docs/literature/ that review no source


def _feed(
    title: str,
    path: str,
    site_url: str,
    entries: list[tuple[str, str, str, str]],
    index: str = "research/index.html",
) -> str:
    """An Atom feed of ``entries``, each (updated, title, href, summary), newest first.

    Args:
        title: The feed's title.
        path: The feed's own path on the site.
        site_url: The site's URL, with a trailing slash.
        entries: The entries, in any order.
        index: The page of the section that the feed follows.

    Returns:
        The feed's XML.
    """
    entries = sorted(entries, key=lambda e: (e[0], e[2]), reverse=True)
    body = []
    for updated, name, href, summary in entries:
        url = escape(site_url + href)
        body.append(
            f"  <entry>\n    <title>{escape(name)}</title>\n"
            f'    <link href="{url}"/>\n    <id>{url}</id>\n'
            f"    <updated>{updated}</updated>\n"
            f"    <summary>{escape(summary)}</summary>\n"
            "  </entry>\n"
        )
    feed_url = escape(site_url + path)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        f"  <title>{escape(title)}</title>\n"
        f'  <link href="{escape(site_url + index)}"/>\n'
        f'  <link rel="self" href="{feed_url}"/>\n'
        f"  <id>{feed_url}</id>\n"
        f"  <updated>{entries[0][0] if entries else NO_DATE}</updated>\n"
        f"{''.join(body)}</feed>\n"
    )


def papers_feed(papers: Sequence, site_url: str, site_title: str) -> str:
    """The feed of the papers (:class:`~openinterpretability.builder.research.Paper`)."""
    entries = [
        (f"{p.published}T00:00:00Z" if p.published else NO_DATE, p.title,
         f"research/{p.href}", p.summary)
        for p in papers
    ]  # fmt: skip
    return _feed(f"{site_title}: papers", PAPERS_FEED, site_url, entries)


def published_feed(papers: Sequence, site_url: str, site_title: str) -> str:
    """The feed of the published papers only, each dated by its publication (#4463).

    A preprint is left out, so a paper enters this feed on the day it is published.
    """
    entries = [
        (f"{p.status.published}T00:00:00Z", p.title, f"research/{p.href}", p.summary)
        for p in papers if p.status is not None and not p.status.preprint
    ]  # fmt: skip
    return _feed(f"{site_title}: published papers", PUBLISHED_FEED, site_url, entries)


def posts_feed(posts: Sequence, site_url: str, site_title: str) -> str:
    """The feed of the posts, each dated by its line in ``posts/README.md`` (#4650).

    Args:
        posts: The posts, as :class:`~openinterpretability.builder.research.Paper`.
        site_url: The site's URL, with a trailing slash.
        site_title: The site's name, which starts the feed's title.

    Returns:
        The feed's XML.

    >>> from types import SimpleNamespace as Post
    >>> post = Post(published="2026-11-20", title="Hello", href="hello/index.html", summary="Hi.")
    >>> xml = posts_feed([post], "https://example.org/", "Site")
    >>> "<updated>2026-11-20T00:00:00Z</updated>" in xml, "example.org/posts/index.html" in xml
    (True, True)
    """
    entries = [
        (f"{p.published}T00:00:00Z" if p.published else NO_DATE, p.title,
         f"posts/{p.href}", p.summary)
        for p in posts
    ]  # fmt: skip
    return _feed(f"{site_title}: posts", POSTS_FEED, site_url, entries, "posts/index.html")


def added_times(folder: Path) -> dict[str, str]:
    """When each Markdown file in ``folder`` was first added, by name without ``.md``.

    One ``git log`` reads every file. It is empty outside a repository or in a shallow clone,
    whose history is incomplete.
    """

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else ""

    if git("rev-parse", "--is-shallow-repository") != "false":
        return {}
    log = git("log", "--diff-filter=A", "--name-only", "--format=@%ct", "--", ".")
    times: dict[str, str] = {}
    when = None
    for line in log.splitlines():  # newest first, so the last time seen is the first add
        if line.startswith("@"):
            stamp = datetime.fromtimestamp(int(line[1:]), UTC)
            when = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif line.endswith(".md") and when:
            times[Path(line).stem] = when
    return times


def literature_feed(reviews: Sequence, folder: Path, site_url: str, site_title: str) -> str:
    """The feed of the reviews (:class:`~openinterpretability.builder.literature_pages.Review`)."""
    rows, times = review_rows(folder), added_times(folder)
    entries = []
    for review in reviews:
        if review.name in NOT_REVIEWS:
            continue
        status, why = rows.get(review.name, ("", ""))
        summary = f"{status.capitalize()}: {why}" if status else ""
        entries.append((times.get(review.name, NO_DATE), review.title, review.href, summary))
    return _feed(f"{site_title}: literature reviews", LITERATURE_FEED, site_url, entries)


def write_research_feeds(output_dir: Path, papers, reviews, folder: Path, common: dict) -> None:
    """Write :data:`PAPERS_FEED`, :data:`PUBLISHED_FEED` and :data:`LITERATURE_FEED`."""
    url, title = common["site_url"], common["site_title"]
    out = Path(output_dir)
    (out / "research").mkdir(parents=True, exist_ok=True)
    (out / PAPERS_FEED).write_text(papers_feed(papers, url, title), encoding="utf-8")
    (out / PUBLISHED_FEED).write_text(published_feed(papers, url, title), encoding="utf-8")
    if reviews:
        text = literature_feed(reviews, folder, url, title)
        (out / LITERATURE_FEED).write_text(text, encoding="utf-8")
