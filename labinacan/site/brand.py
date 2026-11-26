"""The brand's files on the site: the mark, the favicon and the social image (issue #3289).

The brand is the project's, in its config (``labinacan.config``, ``[tool.labinacan.brand]``,
#6629). The files live in the folder that the config names, inside the project's package.
``mark.svg`` is the mark and the favicon, and ``logo.svg`` adds the name. Both are drawn
here from the config's cells and colours (:func:`mark_svg`, :func:`logo_svg`, #4465), and a
project's test checks its committed files. ``mark-32.png`` is the favicon for browsers
without SVG icons. ``og.png`` is the image that a shared link shows. A project's script
draws the PNGs from the mark. Each build copies the files to ``assets/`` (:func:`write_brand`).
"""

from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path

from labinacan.config import config

FILES = ("mark.svg", "mark-32.png", "og.png")
ASSETS = "assets"
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"


def source() -> Path:
    """The folder of the brand's files: the config's ``folder``, inside the package.

    Returns:
        The folder.

    Examples:
        >>> (source() / "mark.svg").exists()
        True
    """
    c = config()
    return Path(str(files(c.package))) / c.brand.folder


def write_brand(output_dir: Path) -> list[Path]:
    """Copy the brand's files into ``output_dir/assets``.

    Args:
        output_dir: The site's folder.

    Returns:
        The paths of the copies, one for each of :data:`FILES`.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     [p.name for p in write_brand(Path(tmp))]
        ['mark.svg', 'mark-32.png', 'og.png']
    """
    target = output_dir / ASSETS
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for name in FILES:
        shutil.copyfile(source() / name, target / name)
        out.append(target / name)
    return out


def _cells(indent: str) -> list[str]:
    """The mark's tile and cells, as SVG elements.

    Args:
        indent: The text before each element.

    Returns:
        One line for each element.
    """
    brand = config().brand
    out = [f'{indent}<rect width="32" height="32" rx="7" fill="{brand.tile}"/>']
    for col, row, share, colour in brand.cells:
        x, y = 5 + 8 * col, 5 + 8 * row
        out.append(f'{indent}<rect x="{x}" y="{y}" width="6" height="6" rx="1" '
                   'fill="#ffffff" fill-opacity="0.16"/>')  # fmt: skip
        if share:  # a share is in thirds (config.brand_of), so 6 * share is whole
            h = 6 * share
            top = y + 6 - h
            out.append(f'{indent}<rect x="{x}" y="{top:g}" width="6" height="{h:g}" '
                       f'rx="1" fill="{brand.colours[colour]}"/>')  # fmt: skip
    return out


def mark_svg() -> str:
    """The mark and favicon, from the brand's cells and colours.

    Returns:
        The SVG document.

    Examples:
        >>> mark_svg().startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"')
        True
    """
    head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" role="img" '
            f'aria-label="{config().name}">')  # fmt: skip
    return "\n".join([head, *_cells("  "), "</svg>"]) + "\n"


def logo_svg() -> str:
    """The mark with the name beside it, the name in the brand's ink.

    Returns:
        The SVG document.

    Examples:
        >>> f">{config().name}</text>" in logo_svg()
        True
    """
    ink, name = config().brand.ink, config().name
    style = (f"  <style>.name {{ fill: {ink['light']}; }} @media (prefers-color-scheme: dark) "
             f"{{ .name {{ fill: {ink['dark']}; }} }}</style>")  # fmt: skip
    text = (f'  <text x="48" y="28" font-family="{FONT}" font-size="21" font-weight="600" '
            f'class="name">{name}</text>')  # fmt: skip
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 330 40" role="img" '
             f'aria-label="{name}">', style, '  <g transform="translate(4 4)">',
             *_cells("    "), "  </g>", text, "</svg>"]  # fmt: skip
    return "\n".join(lines) + "\n"


def write_svgs(folder: Path | None = None) -> None:
    """Write ``mark.svg`` and ``logo.svg`` from the brand.

    Args:
        folder: Where to write them. By default, :func:`source`.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     write_svgs(Path(tmp))
        ...     sorted(p.name for p in Path(tmp).iterdir())
        ['logo.svg', 'mark.svg']
    """
    folder = folder or source()
    (folder / "mark.svg").write_text(
        mark_svg(),
        encoding="utf-8",  # pragma: no mutate, equivalent: the text is ASCII
    )
    (folder / "logo.svg").write_text(
        logo_svg(),
        encoding="utf-8",  # pragma: no mutate, equivalent: the text is ASCII
    )
