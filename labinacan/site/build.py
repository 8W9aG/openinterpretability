"""Build the site of a project that has no build of its own (#6631).

``python -m labinacan.site.build --output-dir dist`` writes the home page and the project's
own pages. Then it writes Research, Posts, the literature review, Changelog, the brand's
images and the feeds.
``make build`` then adds the stylesheet and the search index (``labinacan.mk``).

A project's ``templates/`` folder comes first, so its ``base.html`` and ``index.html`` take
the place of labinacan's defaults. Each other template in that folder is a page of its own,
at the same path. A project with no folder still builds, from the defaults only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import (
    ChoiceLoader,
    Environment,
    FileSystemLoader,
    PackageLoader,
    StrictUndefined,
    select_autoescape,
)

from labinacan.config import config, project_links
from labinacan.site.brand import write_brand
from labinacan.site.feeds import write_site_files
from labinacan.site.render import Renderer, number_filters
from labinacan.site.sections import render_sections

DEFAULT_PAGES = ("index.html",)  # the pages that labinacan writes when a project has none


@dataclass
class SiteReport:
    """What a build wrote."""

    output_dir: Path
    pages: list[Path] = field(default_factory=list)


def environment(templates: Path | None) -> Environment:
    """The templates' environment: the project's folder first, then labinacan's.

    Args:
        templates: The project's templates, or None for labinacan's only.

    Returns:
        The environment, with labinacan's number filters.

    Examples:
        >>> environment(None).get_template("base.html").name
        'base.html'
    """
    loaders = [FileSystemLoader(templates)] if templates and templates.is_dir() else []
    # "templates" is PackageLoader's default folder, and macOS finds "TEMPLATES" too.
    ours = PackageLoader("labinacan.site", "templates")  # pragma: no mutate, equivalent: default
    env = Environment(
        loader=ChoiceLoader([*loaders, ours]),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return number_filters(env)


def own_pages(templates: Path | None) -> list[str]:
    """The project's own pages: each template in its folder that isn't a part of another.

    A template whose name starts with ``_``, and ``base.html``, are parts of other pages.

    Args:
        templates: The project's templates, or None.

    Returns:
        The pages' paths under the site, in order, always with ``index.html``.

    Examples:
        >>> own_pages(None)
        ['index.html']
    """
    names = set(DEFAULT_PAGES)
    if templates and templates.is_dir():
        for path in templates.rglob("*.html"):
            name = path.relative_to(templates).as_posix()
            if not path.name.startswith("_") and name != "base.html":
                names.add(name)
    return sorted(names)


def build(
    output_dir: Path,
    templates: Path | None = None,
    papers_dir: Path | None = None,
    posts_dir: Path | None = None,
    pdfs: bool = False,
) -> SiteReport:
    """Write the site into ``output_dir``.

    Args:
        output_dir: The site's folder.
        templates: The project's templates, or None for labinacan's only.
        papers_dir: The papers, or None for an empty Research section.
        posts_dir: The posts, or None for an empty Posts section.
        pdfs: Print each paper's PDF, which needs headless Chrome.

    Returns:
        What the build wrote.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = SiteReport(output_dir)
    common = {**project_links(), "site_title": config().name, "version": _version()}
    render = Renderer(environment(templates), output_dir, common, report)

    def pages(render) -> None:
        """Write the project's own pages, and the brand's images.

        Args:
            render: The site's renderer.
        """
        for page in own_pages(templates):
            render(page, page)
        if config().brand is not None:  # a project with no brand has no mark to draw
            report.pages.extend(write_brand(output_dir))

    render_sections(render, report, (pages,), papers_dir=papers_dir, posts_dir=posts_dir,
                    pdfs=pdfs, research_index={"explained": False})  # fmt: skip
    write_site_files(output_dir, report.pages, common["site_url"])
    return report


def _version() -> str:
    """The project's version, from its installed distribution, or "" when it isn't one."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(config().command)
    except PackageNotFoundError:
        return ""


def main(argv: list[str] | None = None) -> None:
    """Build the site from the command line.

    Args:
        argv: The arguments, or None for the command line's.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--templates", type=Path, default=Path("templates"))
    parser.add_argument("--papers", type=Path, default=Path("papers"))
    parser.add_argument("--posts", type=Path, default=Path("posts"))
    parser.add_argument("--pdfs", action="store_true", help="Print each paper's PDF.")
    args = parser.parse_args(argv)
    report = build(
        args.output_dir,
        args.templates,
        args.papers if args.papers.is_dir() else None,
        args.posts if args.posts.is_dir() else None,
        args.pdfs,
    )
    print(f"Wrote {len(report.pages)} pages to {args.output_dir}")


if __name__ == "__main__":
    main()
