"""The brand's files on the site: the mark, the favicon and the social image (issue #3289).

The sources live in ``openinterpretability/builder/brand/``. ``mark.svg`` is the mark and
the favicon, and ``logo.svg`` adds the name. Both are drawn here from ``palette.py``
(:func:`mark_svg`, :func:`logo_svg`, #4465), and a test checks the committed files.
``mark-32.png`` is the favicon for browsers without SVG icons. ``og.png`` is the image that
a shared link shows. ``scripts/brand_images.py`` draws the PNGs from the mark.
Each build copies them to ``assets/`` (:func:`write_brand`).
"""

from __future__ import annotations

import shutil
from pathlib import Path

SOURCE = Path(__file__).parent / "brand"
FILES = ("mark.svg", "mark-32.png", "og.png")
ASSETS = "assets"


def write_brand(output_dir: Path) -> list[Path]:
    """Copy the brand's files into ``output_dir/assets``, and return their paths."""
    target = output_dir / ASSETS
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for name in FILES:
        shutil.copyfile(SOURCE / name, target / name)
        out.append(target / name)
    return out


# The mark: a 3×3 layer map, as the site's model pages draw one. Each cell is
# (column, row, share filled, colour); an empty cell is a faint square.
CELLS = (
    (0, 0, 1.0, "yellow"), (1, 0, 1 / 3, "blue"), (2, 0, 2 / 3, "orange"),
    (0, 1, 1 / 3, "orange"), (1, 1, 1.0, "blue"), (2, 1, 0.0, ""),
    (0, 2, 2 / 3, "blue"), (1, 2, 0.0, ""), (2, 2, 1.0, "orange"),
)  # fmt: skip
NAME = "Open Interpretability"
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"


def _cells(indent: str) -> list[str]:
    from openinterpretability.builder.palette import MARK_HEX, MARK_TILE

    out = [f'{indent}<rect width="32" height="32" rx="7" fill="{MARK_TILE}"/>']
    for col, row, share, colour in CELLS:
        x, y = 5 + 8 * col, 5 + 8 * row
        out.append(f'{indent}<rect x="{x}" y="{y}" width="6" height="6" rx="1" '
                   'fill="#ffffff" fill-opacity="0.16"/>')  # fmt: skip
        if share:
            h, top = round(6 * share, 2), round(y + 6 - 6 * share, 2)
            out.append(f'{indent}<rect x="{x}" y="{top:g}" width="6" height="{h:g}" '
                       f'rx="1" fill="{MARK_HEX[colour]}"/>')  # fmt: skip
    return out


def mark_svg() -> str:
    """The mark and favicon, from the palette's colours."""
    head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" role="img" '
            f'aria-label="{NAME}">')  # fmt: skip
    return "\n".join([head, *_cells("  "), "</svg>"]) + "\n"


def logo_svg() -> str:
    """The mark with the name beside it, the name in the page's ink."""
    from openinterpretability.builder.palette import INK

    style = (f"  <style>.name {{ fill: {INK['light']}; }} @media (prefers-color-scheme: dark) "
             f"{{ .name {{ fill: {INK['dark']}; }} }}</style>")  # fmt: skip
    text = (f'  <text x="48" y="28" font-family="{FONT}" font-size="21" font-weight="600" '
            f'class="name">{NAME}</text>')  # fmt: skip
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 330 40" role="img" '
             f'aria-label="{NAME}">', style, '  <g transform="translate(4 4)">',
             *_cells("    "), "  </g>", text, "</svg>"]  # fmt: skip
    return "\n".join(lines) + "\n"


def write_svgs(folder: Path = SOURCE) -> None:
    """Write ``mark.svg`` and ``logo.svg`` from the palette."""
    (folder / "mark.svg").write_text(mark_svg(), encoding="utf-8")
    (folder / "logo.svg").write_text(logo_svg(), encoding="utf-8")
