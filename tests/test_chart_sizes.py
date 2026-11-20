"""Every chart with text draws at its natural size, so its text never scales (#5660)."""

import re
from pathlib import Path

TEMPLATES = Path(__file__).parents[1] / "openinterpretability/builder/templates"

# SVGs that may still scale with the page. The three maps fill the width on purpose. Their
# hover cards are placed in the drawn box (#5666). The pieces grid of a model card has no
# text. It sits inside a link, where a scroll box would nest one control in another.
SCALES = {
    "_concept_map.html": 1,
    "atlas.html": 1,
    "literature.html": 1,
    "_macros.html": 1,
}


def test_no_chart_svg_scales_with_the_page_outside_the_exceptions():
    scaling = re.compile(r'<svg viewBox="[^"]*"[^>]*class="[^"]*\bw-full\b')
    found = {
        path.name: len(scaling.findall(path.read_text(encoding="utf-8")))
        for path in TEMPLATES.rglob("*.html")
    }
    assert {name: n for name, n in found.items() if n} == SCALES
