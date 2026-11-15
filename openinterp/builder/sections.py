"""Pages outside the model tree that every build writes: About and Research."""

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path

from openinterp import DISTRIBUTION
from openinterp.builder.research import copy_assets, load_papers
from openinterp.core.licenses import DATA_TERMS
from openinterp.core.schemas import ReviewPolicy


def render_reference_pages(render, papers_dir: Path | None, report) -> None:
    """Write the About page and the Research section."""
    license = metadata(DISTRIBUTION)["License-Expression"]
    render("about.html", "about.html", policy=ReviewPolicy(), license=license, **DATA_TERMS)
    _render_research(render, papers_dir, report)


def _render_research(render, papers_dir: Path | None, report) -> None:
    """Write the Research section: its index, each paper, and their media."""
    papers = load_papers(papers_dir, render.common["repo_url"])
    render("research/index.html", "research.html", papers=papers)
    for paper in papers:
        report.pages.extend(copy_assets(papers_dir, paper, render.output_dir))
        if not paper.interactive:
            render(f"research/{paper.href}", "paper.html", paper=paper)
