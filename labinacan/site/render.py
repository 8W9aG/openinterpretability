"""The site's renderer, its number filters, and redirect pages (#6628).

A project gives the renderer its Jinja environment, the context that every page gets and a
report that collects the pages written. It owns some paths in the output folder, which
:func:`clean_owned` removes before each build.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment

from labinacan.site.contents import add_contents


def root_for(rel: str) -> str:
    """The relative path from the page at ``rel`` back to the site's root.

    Args:
        rel: The page's path under the site, such as ``models/m/index.html``.

    Returns:
        ``../`` once for each folder above the page.

    Examples:
        >>> root_for("index.html"), root_for("models/m/index.html")
        ('', '../../')
    """
    return "../" * (len(Path(rel).parts) - 1)


@dataclass
class Renderer:
    """Renders a template to a path under the output folder, with the site-wide context."""

    env: Environment
    output_dir: Path
    common: dict
    report: Any  # what the build wrote: it has a ``pages`` list
    jobs: int = 1  # worker processes for the unit pages (labinacan/site/parallel.py)

    def __call__(self, rel: str, template: str, **context) -> None:
        """Write the page at ``rel`` from ``template``, and record it in the report.

        Args:
            rel: The page's path under the output folder.
            template: The template's name.
            **context: What the template gets besides :attr:`common`.
        """
        out = self.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        page = self.env.get_template(template)
        url = self.common["site_url"] + rel
        html = page.render(**self.common, root=root_for(rel), canonical=url, **context)
        out.write_text(add_contents(html))  # ids on headings, and contents on a long page
        self.report.pages.append(out)


def write_redirect(out: Path, target: str, report: Any) -> None:
    """Write a page that used to live at ``out`` and now lives at ``target``.

    Args:
        out: The old page's file.
        target: The new page, relative to the old one.
        report: What the build wrote: it has a ``pages`` list.

    Examples:
        >>> import tempfile, types
        >>> report = types.SimpleNamespace(pages=[])
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     write_redirect(Path(tmp) / "a" / "old.html", "new.html", report)
        ...     'url=new.html' in (Path(tmp) / "a" / "old.html").read_text()
        True
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<!doctype html><meta charset="utf-8"><title>Moved</title>'
        f'<link rel="canonical" href="{target}"><meta name="robots" content="noindex">'
        f'<meta http-equiv="refresh" content="0; url={target}">'
        f'<p>This page moved to <a href="{target}">{target}</a>.</p>\n'
    )
    report.pages.append(out)


def clean_owned(output_dir: Path, names: Iterable[str]) -> None:
    """Remove each path that the build owns in ``output_dir``, so no stale page is left.

    Args:
        output_dir: The site's output folder.
        names: The files and folders that the build writes, by their names in it.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = Path(tmp)
        ...     (out / "models").mkdir()
        ...     _ = (out / "index.html").write_text("x"), (out / "keep.txt").write_text("y")
        ...     clean_owned(out, ("models", "index.html", "gone"))
        ...     sorted(p.name for p in out.iterdir())
        ['keep.txt']
    """
    for name in names:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def pct_fine(v: float | None) -> str:
    """A signed percentage with two significant figures, so 1 in 100k isn't shown as 0%.

    Args:
        v: A share, such as ``0.021``, or None.

    Returns:
        The percentage, or "—" for None.

    Examples:
        >>> pct_fine(0.021), pct_fine(0), pct_fine(None)
        ('2.1%', '0%', '—')
    """
    if v is None:
        return "—"
    if v == 0:
        return "0%"
    decimals = max(0, 1 - math.floor(math.log10(100 * abs(v))))
    return f"{100 * v:.{decimals}f}%"


def number_filters(env: Environment) -> Environment:
    """Add the number filters that the pages use, such as ``pct`` and ``pval``.

    They are ``pct``, ``pct_fine``, ``num``, ``thousands`` and ``pval``.

    Args:
        env: The templates' environment.

    Returns:
        ``env``, with the filters.

    Examples:
        >>> f = number_filters(Environment()).filters
        >>> f["pct"](0.5), f["num"](0.12345), f["thousands"](1234), f["pval"](0.0001)
        ('50%', '0.123', '1,234', '1.0e-04')
    """
    env.filters["pct"] = lambda v: "—" if v is None else f"{100 * v:.0f}%"
    env.filters["pct_fine"] = pct_fine
    env.filters["num"] = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    env.filters["thousands"] = lambda v: "?" if v is None else f"{v:,}"
    # p-values below 0.001 would round to 0.000, so show them in scientific notation.
    env.filters["pval"] = lambda v: "—" if v is None else f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"
    return env
