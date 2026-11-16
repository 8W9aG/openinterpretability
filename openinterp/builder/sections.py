"""Pages outside the model tree that every build writes: About and Research."""

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path

from openinterp import DISTRIBUTION
from openinterp.builder.research import copy_assets, load_papers
from openinterp.core.licenses import DATA_TERMS
from openinterp.core.schemas import ReviewPolicy


def render_reference_pages(render, papers_dir: Path | None, report, *, pdfs: bool = False) -> None:
    """Write the About page and the Research section, with a PDF of each paper if ``pdfs``."""
    license = metadata(DISTRIBUTION)["License-Expression"]
    render("about.html", "about.html", policy=ReviewPolicy(), license=license, **DATA_TERMS)
    _render_research(render, papers_dir, report, pdfs)


def _render_research(render, papers_dir: Path | None, report, pdfs: bool) -> None:
    """Write the Research section: its index, each paper, their media and their PDFs.

    A paper's PDF is printed before its page, so the page links it only if it exists.
    """
    from openinterp.builder.paper_pdf import chrome, write_pdf

    papers = load_papers(papers_dir, render.common["repo_url"])
    render("research/index.html", "research.html", papers=papers)
    browser = chrome() if pdfs else None
    if pdfs and browser is None:
        print("No Chrome found ($OPENINTERP_CHROME): the papers get no PDFs.")
    for paper in papers:
        report.pages.extend(copy_assets(papers_dir, paper, render.output_dir))
        if not paper.interactive:
            folder = render.output_dir / "research" / paper.slug
            folder.mkdir(parents=True, exist_ok=True)
            pdf = write_pdf(render.env, folder, paper, render.common, browser)
            render(f"research/{paper.href}", "paper.html", paper=paper, pdf=pdf)
