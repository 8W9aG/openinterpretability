"""Pages outside the model tree that every build writes: About, Changelog, Research and Posts."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from importlib.metadata import metadata
from pathlib import Path

from labinacan.site.brand import write_brand
from labinacan.site.sections import render_sections
from openinterpretability import DISTRIBUTION
from openinterpretability.builder.explanation_pages import render_explanations
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

    labinacan writes the sections that any project has (:func:`render_sections`). This
    repo adds About, with the brand's images, and the explanations (#6628).

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
    views = list(explanations)

    def about(render) -> None:
        """Write About, and the mark, favicon and social image.

        Args:
            render: The site's renderer.
        """
        license = metadata(DISTRIBUTION)["License-Expression"]
        taxonomy = taxonomy_rows(render.common["repo_url"])
        render("about.html", "about.html", policy=ReviewPolicy(), license=license,
               taxonomy=taxonomy, **DATA_TERMS)  # fmt: skip
        report.pages.extend(write_brand(render.output_dir))

    def explained(render) -> None:
        """Write the explanations' pages.

        Args:
            render: The site's renderer.
        """
        render_explanations(render, views, unit_pages)

    index = {"explained": bool(views)}  # the Research index links the explanations
    render_sections(render, report, (about, explained), papers_dir=papers_dir,
                    posts_dir=posts_dir, pdfs=pdfs, research_index=index)  # fmt: skip
