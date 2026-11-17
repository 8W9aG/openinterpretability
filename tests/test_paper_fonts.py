"""The paper PDFs are set in Computer Modern, with the fonts embedded (paper_pdf.py, #4307)."""

import base64
import re

from openinterpretability.builder.compiler import _environment
from openinterpretability.builder.paper_pdf import (
    FONT_FACES,
    FONTS,
    MATH_FACE,
    font_faces,
    print_body,
    print_svg,
    raise_superscripts,
)

SYSTEM_FONTS = ("Times", "Liberation", "Nimbus", "Courier", "Menlo", "Helvetica", "Arial")


def test_every_bundled_font_and_its_licence_is_there():
    for _, _, _, name in (*FONT_FACES, MATH_FACE):
        assert (FONTS / name).stat().st_size > 10_000, name
    assert "SIL Open Font License" in (FONTS / "OFL.txt").read_text()
    assert "GUST" in (FONTS / "GUST-FONT-LICENSE.txt").read_text()


def test_the_print_page_names_only_the_bundled_faces():
    source = (_environment().loader.get_source(_environment(), "paper_print.html"))[0]
    declarations = re.findall(r"font(?:-family)?:([^;}]*)", source)
    families = {name for d in declarations for name in re.findall(r'"([^"]+)"', d)}
    assert families == {"CMU Serif", "CMU Typewriter Text", "Latin Modern Math"}
    assert not any(name in source for name in SYSTEM_FONTS)


def test_each_face_is_a_data_uri():
    faces = font_faces()
    assert faces.count("@font-face") == len(FONT_FACES) + 1
    assert faces.count("data:font/woff2;base64,") == len(FONT_FACES)
    assert "data:font/otf;base64," in faces and "http" not in faces
    assert font_faces(math=False).count("@font-face") == len(FONT_FACES)


def test_a_figure_gets_the_fonts_and_the_maths_face_only_when_it_needs_it(tmp_path):
    plain = '<svg xmlns="http://www.w3.org/2000/svg"><text>a</text></svg>'
    assert 'Latin Modern Math"; font-weight' not in print_svg(plain)
    assert '"CMU Serif"; font-weight: 400' in print_svg(plain)
    arrows = '<svg xmlns="http://www.w3.org/2000/svg"><text>a ↔ b</text></svg>'
    assert 'font-family: "Latin Modern Math"; font-weight' in print_svg(arrows)
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "f.svg").write_text(plain)
    body = print_body('<p><img src="media/f.svg" alt="x"> a � b</p>', tmp_path)
    (uri,) = re.findall(r'src="data:image/svg\+xml;base64,([^"]+)"', body)
    assert "CMU Serif" in base64.b64decode(uri).decode()
    assert "\\ufffd" in body and "�" not in body


def test_superscripts_are_raised_in_text_but_not_in_attributes():
    html = '<p title="10⁻⁶">so (10⁻⁶) and x²</p>'
    assert raise_superscripts(html) == '<p title="10⁻⁶">so (10<sup>−6</sup>) and x<sup>2</sup></p>'
    svg = raise_superscripts('<text aria-label="10⁻⁶">10⁻⁶</text>', svg=True)
    assert svg.startswith('<text aria-label="10⁻⁶">10<tspan baseline-shift="super"')
