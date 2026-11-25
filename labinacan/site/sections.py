"""The sections that any project's site has: Changelog, the literature review, Research, Posts.

A project adds its own pages, such as an About page, as callbacks (:data:`Section`). Each
one gets the renderer and writes its pages (#6628).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from labinacan.site.changelog_page import load_versions
from labinacan.site.literature_pages import LITERATURE, load_reviews, render_reviews
from labinacan.site.research import copy_assets, load_papers
from labinacan.site.research_feeds import POSTS_FEED, posts_feed, write_research_feeds

Section = Callable[[Any], None]
"""A project's own section: a callback that writes its pages with the renderer it gets."""


def render_sections(
    render,
    report,
    own: Sequence[Section] = (),
    *,
    papers_dir: Path | None = None,
    posts_dir: Path | None = None,
    pdfs: bool = False,
    research_index: dict | None = None,
) -> None:
    """Write the project's own sections, then Changelog, the reviews, Research and Posts.

    Args:
        render: The site's renderer.
        report: The build's report, which collects the pages written.
        own: The project's own sections, written first, in order.
        papers_dir: The folder of the papers, or None for an empty Research section.
        posts_dir: The folder of the posts, or None for an empty Posts section (#4650).
        pdfs: Print each paper's PDF.
        research_index: What else the Research index gets. It always gets ``reviewed``:
            whether the literature review has pages.
    """
    for section in own:
        section(render)
    versions = load_versions(render.common["repo_url"])
    if versions:  # an install has no CHANGELOG.md
        render("changelog.html", "changelog.html", versions=versions)
    reviews = load_reviews(render.common["repo_url"])
    render_reviews(render, reviews)
    index = {**(research_index or {}), "reviewed": bool(reviews)}
    papers = render_research(render, papers_dir, report, pdfs, index)
    write_research_feeds(render.output_dir, papers, reviews, LITERATURE, render.common)
    render_posts(render, posts_dir, report)


def render_posts(render, posts_dir: Path | None, report) -> None:
    """Write the Posts section: its index, each post, their media and their feed (#4650).

    A post has the form of a paper (``papers/``), so the paper loader reads it, with its
    Markdown, maths and redaction. Only the section folder and the templates differ. A draft
    (#5529, #5551) builds at its own URL under ``posts/drafts/``, and no index, feed or
    sitemap lists it.

    Args:
        render: The site's renderer.
        posts_dir: The folder of the posts, or None for an index with no posts.
        report: The build's report, which collects the pages written.
    """
    posts = load_papers(posts_dir, render.common["repo_url"], source="posts")
    listed = [post for post in posts if not post.draft]  # a draft is built, never listed (#5551)
    render("posts/index.html", "posts.html", posts=listed)
    for post in posts:
        report.pages.extend(copy_assets(posts_dir, post, render.output_dir, section="posts"))
        if not post.interactive:
            render(f"posts/{post.href}", "post.html", post=post)
    feed = posts_feed(listed, render.common["site_url"], render.common["site_title"])
    (render.output_dir / POSTS_FEED).write_bytes(feed.encode())  # UTF-8 on any locale


def render_research(render, papers_dir: Path | None, report, pdfs: bool, index: dict) -> list:
    """Write the Research section: its index, each paper, their media and their PDFs.

    A paper's PDF is printed before its page, so the page links it only if it exists.

    Args:
        render: The site's renderer.
        papers_dir: The folder of the papers, or None for an index with no papers.
        report: The build's report, which collects the pages written.
        pdfs: Print each paper's PDF.
        index: What else the index template gets, such as which sections exist.

    Returns:
        The papers, for the section's feed.
    """
    from labinacan.site.paper_pdf import CHROME_ENV, chrome, write_pdf

    papers = load_papers(papers_dir, render.common["repo_url"])
    render("research/index.html", "research.html", papers=papers, **index)
    browser = chrome() if pdfs else None
    if pdfs and browser is None:
        print(f"No Chrome found (${CHROME_ENV}): the papers get no PDFs.")
    for paper in papers:
        report.pages.extend(copy_assets(papers_dir, paper, render.output_dir))
        if not paper.interactive:
            folder = render.output_dir / "research" / paper.slug
            folder.mkdir(parents=True, exist_ok=True)
            pdf = write_pdf(render.env, folder, paper, render.common, browser)
            render(f"research/{paper.href}", "paper.html", paper=paper, pdf=pdf)
    return papers
