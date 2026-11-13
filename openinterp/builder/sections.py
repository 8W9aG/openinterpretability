"""Pages outside the model tree that every build writes: About, Errata and Research."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import metadata
from pathlib import Path

from openinterp.builder.research import copy_assets, load_papers
from openinterp.core.errata import Erratum
from openinterp.core.licenses import DATA_TERMS
from openinterp.core.schemas import ReviewPolicy


def render_reference_pages(render, models, errata: Sequence[Erratum], papers_dir, report) -> None:
    """Write the About page, the Errata page and the Research section."""
    license = metadata("openinterp")["License-Expression"]
    render("about.html", "about.html", policy=ReviewPolicy(), license=license, **DATA_TERMS)
    render("errata.html", "errata.html", errata=errata_view(errata, models))
    _render_research(render, papers_dir, report)


def errata_view(errata: Sequence[Erratum], models) -> list[dict]:
    """Each erratum with the experiments it covers, for the Errata page, newest first."""
    covered: dict[str, list[dict]] = {erratum.id: [] for erratum in errata}
    for m in models:
        for layer in m.layers:
            for n in layer.units:
                for e in n.experiments:
                    for erratum in e.errata:
                        row = {"model": m, "layer": layer, "unit": n, "e": e}
                        covered[erratum.id].append(row)
    ordered = sorted(errata, key=lambda erratum: erratum.date, reverse=True)
    return [{"erratum": erratum, "covered": covered[erratum.id]} for erratum in ordered]


def _render_research(render, papers_dir: Path | None, report) -> None:
    """Write the Research section: its index, each paper, and their media."""
    papers = load_papers(papers_dir, render.common["repo_url"])
    render("research/index.html", "research.html", papers=papers)
    for paper in papers:
        report.pages.extend(copy_assets(papers_dir, paper, render.output_dir))
        if not paper.interactive:
            render(f"research/{paper.href}", "paper.html", paper=paper)
