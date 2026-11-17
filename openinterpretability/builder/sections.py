"""Pages outside the model tree that every build writes: About, Changelog and Research."""

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path

from openinterpretability import DISTRIBUTION
from openinterpretability.builder.brand import write_brand
from openinterpretability.builder.changelog_page import load_versions
from openinterpretability.builder.explanation_pages import render_explanations
from openinterpretability.builder.research import copy_assets, load_papers
from openinterpretability.core.licenses import DATA_TERMS
from openinterpretability.core.schemas import ReviewPolicy


def render_reference_pages(
    render, papers_dir: Path | None, report, *, pdfs: bool = False, explanations=()
) -> None:
    """Write About, Changelog, the Research section and the explanations.

    ``pdfs`` prints each paper's PDF. ``explanations`` are
    :class:`~openinterpretability.builder.explanation_pages.ExplanationView`.
    """
    license = metadata(DISTRIBUTION)["License-Expression"]
    render("about.html", "about.html", policy=ReviewPolicy(), license=license, **DATA_TERMS)
    report.pages.extend(write_brand(render.output_dir))  # the mark, favicon and social image
    versions = load_versions(render.common["repo_url"])
    if versions:  # an install has no CHANGELOG.md
        render("changelog.html", "changelog.html", versions=versions)
    _render_research(render, papers_dir, report, pdfs, bool(explanations))
    render_explanations(render, list(explanations))


def _render_research(render, papers_dir: Path | None, report, pdfs: bool, explained: bool) -> None:
    """Write the Research section: its index, each paper, their media and their PDFs.

    A paper's PDF is printed before its page, so the page links it only if it exists.
    """
    from openinterpretability.builder.paper_pdf import chrome, write_pdf

    papers = load_papers(papers_dir, render.common["repo_url"])
    render("research/index.html", "research.html", papers=papers, explained=explained)
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
