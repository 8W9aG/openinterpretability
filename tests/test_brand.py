"""The brand: the mark, favicon, social card and one accent colour (#3289).

The code is labinacan/site/brand.py, and the brand is in pyproject.toml (#6629).
"""

import struct
import subprocess
import xml.etree.ElementTree as ET

from labinacan.config import config
from labinacan.site import brand
from openinterpretability.builder import palette
from openinterpretability.builder.compiler import compile_site


def _png_size(path):
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def test_the_brand_files_exist_and_the_images_have_their_sizes():
    from scripts.brand_images import SIZES

    for name in ("mark.svg", "logo.svg"):
        ET.parse(brand.source() / name)  # well-formed SVG
    for name, size in SIZES.items():
        assert _png_size(brand.source() / name) == size
    assert set(brand.FILES) <= {p.name for p in brand.source().iterdir()}


def test_the_site_shows_the_mark_favicon_and_card(tmp_path, record):
    compile_site([record], tmp_path)
    for name in brand.FILES:
        assert (tmp_path / "assets" / name).exists()
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'rel="icon" href="assets/mark.svg"' in page and "⊕" not in page
    assert '<meta property="og:image" content="https://' in page and "assets/og.png" in page
    assert 'name="twitter:card" content="summary_large_image"' in page
    meta = '<meta name="description" content="An agent-run mechanistic interpretability project'
    assert meta in page


def test_the_accent_has_one_name():
    colours = palette.tailwind_colors()
    assert colours["brand-700"] == "#047857" and len(palette.BRAND_HEX) == 11
    found = subprocess.run(
        ["git", "grep", "-l", "emerald-[0-9]", "--", "openinterpretability/builder/templates",
         "labinacan/site/templates"],
        capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip
    assert found == ""  # templates name the accent `brand-`, defined once in palette.py


def test_the_mark_and_logo_are_drawn_from_the_palette():
    # The committed SVGs are the generated ones, so the colours have one source (#4465).
    assert (brand.source() / "mark.svg").read_text(encoding="utf-8") == brand.mark_svg()
    assert (brand.source() / "logo.svg").read_text(encoding="utf-8") == brand.logo_svg()
    used = {c for _, _, share, c in config().brand.cells if share}
    assert used == set(config().brand.colours)
    layouts = {*palette.STATUS_HEX.values(), *palette.POLE_HEX.values()}
    assert set(config().brand.colours.values()) <= layouts
    for hex_colour in config().brand.colours.values():
        assert hex_colour in brand.mark_svg()


def test_the_mark_colours_differ_in_lightness_and_stand_out_on_the_tile():
    import itertools

    from tests.test_palette import contrast, linear, oklab

    lightness = {n: oklab(linear(h))[0] for n, h in config().brand.colours.items()}
    for a, b in itertools.combinations(lightness, 2):
        assert abs(lightness[a] - lightness[b]) >= 0.1, (a, b)  # not hue alone (deuteranopia)
    for name, hex_colour in config().brand.colours.items():
        assert contrast(hex_colour, config().brand.tile) >= 3.0, name


def test_the_header_has_no_tagline(tmp_path):
    from openinterpretability.builder.compiler import compile_site

    compile_site([], tmp_path)
    page = (tmp_path / "index.html").read_text()
    header = page[page.index("<header") : page.index("</header>")]
    assert "interpretability project" not in header  # removed from the header (#5535)
    meta = '<meta name="description" content="An agent-run mechanistic interpretability project'
    assert meta in page


def test_write_svgs_writes_the_mark_and_the_logo_from_the_palette(tmp_path):
    brand.write_svgs(tmp_path)
    assert (tmp_path / "mark.svg").read_text(encoding="utf-8") == brand.mark_svg()
    assert (tmp_path / "logo.svg").read_text(encoding="utf-8") == brand.logo_svg()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["logo.svg", "mark.svg"]


def test_the_mark_is_drawn_from_the_brand_in_the_config():
    b = config().brand
    mark, logo = brand.mark_svg(), brand.logo_svg()
    assert f'aria-label="{config().name}"' in mark and f">{config().name}</text>" in logo
    assert f'fill="{b.tile}"' in mark and b.ink["light"] in logo and b.ink["dark"] in logo
    # Each filled cell's bar has the share of the 6-pixel cell that the config gives.
    assert mark.count('fill-opacity="0.16"') == len(b.cells)
    for thirds in (1, 2, 3):
        bars = [c for _, _, share, c in b.cells if share and round(share * 3) == thirds]
        drawn = sum(
            mark.count(f'height="{2 * thirds}" rx="1" fill="{b.colours[c]}"') for c in set(bars)
        )
        assert drawn == len(bars), thirds


def test_the_brand_can_be_written_again_over_an_earlier_build(tmp_path):
    site = tmp_path / "not" / "made" / "yet"  # the folders are made as needed
    first = brand.write_brand(site)
    assert brand.write_brand(site) == first  # the assets folder may exist already
