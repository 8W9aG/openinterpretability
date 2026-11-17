"""The brand: the mark, favicon, social card and one accent colour (builder/brand.py, #3289)."""

import struct
import subprocess
import xml.etree.ElementTree as ET

from openinterpretability.builder import brand, palette
from openinterpretability.builder.compiler import compile_site


def _png_size(path):
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def test_the_brand_files_exist_and_the_images_have_their_sizes():
    from scripts.brand_images import SIZES

    for name in ("mark.svg", "logo.svg"):
        ET.parse(brand.SOURCE / name)  # well-formed SVG
    for name, size in SIZES.items():
        assert _png_size(brand.SOURCE / name) == size
    assert set(brand.FILES) <= {p.name for p in brand.SOURCE.iterdir()}


def test_the_site_shows_the_mark_favicon_and_card(tmp_path, record):
    compile_site([record], tmp_path)
    for name in brand.FILES:
        assert (tmp_path / "assets" / name).exists()
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'rel="icon" href="assets/mark.svg"' in page and "⊕" not in page
    assert '<meta property="og:image" content="https://' in page and "assets/og.png" in page
    assert 'name="twitter:card" content="summary_large_image"' in page
    assert '<meta name="description" content="An agent-audited' in page


def test_the_accent_has_one_name():
    colours = palette.tailwind_colors()
    assert colours["brand-700"] == "#047857" and len(palette.BRAND_HEX) == 11
    found = subprocess.run(
        ["git", "grep", "-l", "emerald-[0-9]", "--", "openinterpretability/builder/templates"],
        capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip
    assert found == ""  # templates name the accent `brand-`, defined once in palette.py
