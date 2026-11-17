"""The brand's files on the site: the mark, the favicon and the social image (issue #3289).

The sources live in ``openinterpretability/builder/brand/``. ``mark.svg`` is the mark and
the favicon. ``mark-32.png`` is the favicon for browsers without SVG icons. ``og.png`` is the
image that a shared link shows. ``scripts/brand_images.py`` draws the PNGs from the mark.
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
