"""Pages outside the model tree that every build writes: About, Changelog and Research."""

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path

from openinterpretability import DISTRIBUTION
from openinterpretability.builder.brand import write_brand
from openinterpretability.builder.changelog_page import load_versions
from openinterpretability.builder.explanation_pages import render_explanations
from openinterpretability.builder.literature_pages import LITERATURE, load_reviews, render_reviews
from openinterpretability.builder.research import copy_assets, load_papers
from openinterpretability.builder.research_feeds import write_research_feeds
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


def render_reference_pages(
    render, papers_dir: Path | None, report, *, pdfs: bool = False, explanations=()
) -> None:
    """Write About, Changelog, the Research section, its literature review and the explanations.

    ``pdfs`` prints each paper's PDF. ``explanations`` are
    :class:`~openinterpretability.builder.explanation_pages.ExplanationView`.
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
    render_explanations(render, list(explanations))


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
