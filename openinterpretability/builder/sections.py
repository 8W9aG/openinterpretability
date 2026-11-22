"""Pages outside the model tree that every build writes: About, Changelog, Research and Posts."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from importlib.metadata import metadata
from pathlib import Path

from openinterpretability import DISTRIBUTION
from openinterpretability.builder.brand import write_brand
from openinterpretability.builder.changelog_page import load_versions
from openinterpretability.builder.explanation_pages import render_explanations
from openinterpretability.builder.literature_pages import LITERATURE, load_reviews, render_reviews
from openinterpretability.builder.research import copy_assets, load_papers
from openinterpretability.builder.research_feeds import POSTS_FEED, posts_feed, write_research_feeds
from openinterpretability.builder.taxonomy_view import taxonomy_rows
from openinterpretability.core.licenses import DATA_TERMS
from openinterpretability.core.schemas import ReviewPolicy


def with_explanations(stats: dict, explanations) -> dict:
    """``stats`` with the home page's explanation count: one per prompt page (#4574).

    The count reads the same prompt views that :func:`render_explanations` writes, so the
    tile and the Explanations index agree. A second run on a prompt updates its page, so it
    doesn't count again.
    """
    stats["totals"]["explanations"] = len(list(explanations or ()))
    return stats


def stats_for_pages(render, stats: dict, explanations) -> dict:
    """``stats`` for the pages: the explanations' tile, and each model's baseline (#5838).

    A model's page shows its own learned-basis baseline, so ``render.common`` gets the
    baselines by model name.

    Args:
        render: The site's renderer, whose ``common`` context every page gets.
        stats: The registry's stats (``registry_stats``).
        explanations: The explanations' prompt views.

    Returns:
        ``stats``, with the explanation count.

    Examples:
        >>> from types import SimpleNamespace
        >>> render = SimpleNamespace(common={})
        >>> found = stats_for_pages(render, {"totals": {}, "baseline": []}, [1, 2])
        >>> found["totals"]["explanations"], render.common["baselines"]
        (2, {})
    """
    render.common["baselines"] = {b.model_name: b for b in stats["baseline"]}
    return with_explanations(stats, explanations)


def render_reference_pages(
    render,
    papers_dir: Path | None,
    report,
    pdfs: bool = False,
    explanations=(),
    posts_dir: Path | None = None,
    unit_pages: AbstractSet[str] | None = None,
) -> None:
    """Write About, Changelog, Research, its literature review, the explanations and Posts.

    Args:
        render: The site's renderer.
        papers_dir: The folder of the papers, or None for an empty Research section.
        report: The build's report, which collects the pages written.
        pdfs: Print each paper's PDF.
        explanations: The :class:`~openinterpretability.builder.explanation_pages.ExplanationView`
            of each prompt.
        posts_dir: The folder of the posts, or None for an empty Posts section (#4650).
        unit_pages: The unit pages that the build writes, so that an explanation or an
            answer links only those (#5727). None links every unit.
    """
    license = metadata(DISTRIBUTION)["License-Expression"]
    taxonomy = taxonomy_rows(render.common["repo_url"])
    render("about.html", "about.html", policy=ReviewPolicy(), license=license, taxonomy=taxonomy,
           **DATA_TERMS)  # fmt: skip
    report.pages.extend(write_brand(render.output_dir))  # the mark, favicon and social image
    versions = load_versions(render.common["repo_url"])
    if versions:  # an install has no CHANGELOG.md
        render("changelog.html", "changelog.html", versions=versions)
    reviews = load_reviews(render.common["repo_url"])
    render_reviews(render, reviews)
    papers = _render_research(render, papers_dir, report, pdfs, bool(explanations), bool(reviews))
    write_research_feeds(render.output_dir, papers, reviews, LITERATURE, render.common)
    render_explanations(render, list(explanations), unit_pages)
    _render_posts(render, posts_dir, report)


def _render_posts(render, posts_dir: Path | None, report) -> None:
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
    (render.output_dir / POSTS_FEED).write_text(feed, encoding="utf-8")


def _render_research(
    render, papers_dir: Path | None, report, pdfs: bool, explained: bool, reviewed: bool
) -> list:
    """Write the Research section: its index, each paper, their media and their PDFs.

    Return the papers, for the section's feed.

    A paper's PDF is printed before its page, so the page links it only if it exists.
    """
    from openinterpretability.builder.paper_pdf import chrome, write_pdf

    papers = load_papers(papers_dir, render.common["repo_url"])
    render("research/index.html", "research.html", papers=papers, explained=explained,
           reviewed=reviewed)  # fmt: skip
    browser = chrome() if pdfs else None
    if pdfs and browser is None:
        print("No Chrome found ($OPENINTERPRETABILITY_CHROME): the papers get no PDFs.")
    for paper in papers:
        report.pages.extend(copy_assets(papers_dir, paper, render.output_dir))
        if not paper.interactive:
            folder = render.output_dir / "research" / paper.slug
            folder.mkdir(parents=True, exist_ok=True)
            pdf = write_pdf(render.env, folder, paper, render.common, browser)
            render(f"research/{paper.href}", "paper.html", paper=paper, pdf=pdf)
    return papers
