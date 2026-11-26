"""Draw the brand's PNG images from its SVG mark, with headless Chrome (issue #3289).

The mark is ``mark.svg`` in the brand's folder, which the config names
(``[tool.labinacan.brand]``, #6629). This script writes, next to it:

- ``mark-32.png``: the favicon for browsers that don't take an SVG icon;
- ``og.png``: the 1,200 × 630 image that a shared link shows (Open Graph and Twitter cards).
  It is also GitHub's social preview, which the maintainer uploads in the repository's settings.

It first writes ``mark.svg`` and ``logo.svg`` from the brand (``brand.write_svgs``). The
PNGs are committed, so the site build needs no browser. Run this again after a change to
the brand or the project's description::

    uv run python scripts/brand_images.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labinacan.config import config, env, use  # noqa: E402
from labinacan.site.brand import source, write_svgs  # noqa: E402
from labinacan.site.paper_pdf import chrome  # noqa: E402

use(Path(__file__).resolve().parents[1] / "pyproject.toml")  # this repository's values (#6629)

SIZES = {"mark-32.png": (32, 32), "og.png": (1200, 630)}

OG_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html, body {{ margin: 0; width: 1200px; height: 630px; background: #020617; }}
body {{ display: flex; align-items: center; gap: 56px; padding: 0 96px; box-sizing: border-box;
       font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
img {{ width: 220px; height: 220px; flex: none; }}
h1 {{ margin: 0 0 20px; font-size: 64px; font-weight: 650; color: #f8fafc; }}
p {{ margin: 0; font-size: 30px; line-height: 1.35; color: #94a3b8; }}
</style></head><body><img src="{mark}" alt=""><div><h1>{name}</h1>
<p>{tagline}</p></div></body></html>
"""
MARK_PAGE = """<!doctype html><html><head><style>
html, body {{ margin: 0; background: transparent; }}
img {{ display: block; width: 32px; height: 32px; }}</style></head>
<body><img src="{mark}" alt=""></body></html>
"""


def tagline() -> str:
    """The project's description without its leading name: one source for the summary."""
    return config().description.rstrip(".")


def shoot(browser: str, page: str, out: Path, size: tuple[int, int]) -> None:
    """Render the HTML ``page`` at ``size`` and save it as the PNG ``out``."""
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "page.html"
        html.write_text(page, encoding="utf-8")
        width, height = size
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--default-background-color=00000000", f"--window-size={width},{height}",
             f"--screenshot={out}", html.as_uri()],
            check=True, capture_output=True, timeout=60,
        )  # fmt: skip


def main() -> None:
    """Write every PNG in :data:`SIZES` from the mark."""
    write_svgs()  # the SVGs first, from the brand (#4465)
    browser = chrome()
    if browser is None:
        raise SystemExit(f"No Chrome found: set ${env('CHROME')}.")
    brand = source()
    mark = (brand / "mark.svg").as_uri()
    shoot(browser, MARK_PAGE.format(mark=mark), brand / "mark-32.png", SIZES["mark-32.png"])
    og = OG_PAGE.format(mark=mark, name=config().name,
                        tagline=tagline()[0].upper() + tagline()[1:])  # fmt: skip
    shoot(browser, og, brand / "og.png", SIZES["og.png"])
    for name in SIZES:
        print(brand / name)


if __name__ == "__main__":
    main()
