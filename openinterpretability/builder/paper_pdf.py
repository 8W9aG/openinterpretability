"""A PDF of each paper, laid out like a conference paper (issue #3140).

The build writes a print page for the paper (``templates/paper_print.html``), with a
stylesheet in the style of a NeurIPS paper. Headless Chrome prints it to ``paper.pdf``
beside the paper's page, and the print page is then deleted. Chrome reads the figures'
CSS (their colour variables and the transforms of moving marks) as the site does. With
reduced motion forced on, each animated figure prints its resting frame.

A machine with no Chrome builds no PDFs, and the paper pages then show no PDF link.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
from functools import cache
from pathlib import Path

CHROME_ENV = "OPENINTERPRETABILITY_CHROME"
# Where Chrome or Chromium usually is: macOS, then Linux (GitHub's runners have google-chrome).
CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)
CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
FONTS = Path(__file__).parent / "fonts"  # CMU, the Computer Modern Unicode fonts (#4307)
# Each face of the print page: (family, weight, style, file in FONTS).
FONT_FACES = (
    ("CMU Serif", 400, "normal", "cmu-serif-500-roman.woff2"),
    ("CMU Serif", 400, "italic", "cmu-serif-500-italic.woff2"),
    ("CMU Serif", 700, "normal", "cmu-serif-700-roman.woff2"),
    ("CMU Serif", 700, "italic", "cmu-serif-700-italic.woff2"),
    ("CMU Typewriter Text", 400, "normal", "cmu-typewriter-text-500-roman.woff2"),
)
# Computer Modern's maths face, for the symbols that CMU's text faces lack (≈, ↔, ′).
MATH_FACE = ("Latin Modern Math", 400, "normal", "latinmodern-math.otf")
# Maths symbols: arrows, operators, primes and technical signs. A figure with one gets
# the maths face too. It is large, so a figure without one doesn't.
_MATH = re.compile("[\u2032-\u2037\u2190-\u23ff\u27c0-\u27ff\u2900-\u2aff]")
# Superscript digits and signs, which no Computer Modern face has. LaTeX raises the
# plain characters, and so does the print page.
_SUPER = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
_PLAIN = "0123456789+−"
_SUPERS = re.compile(f"[{_SUPER}]+")
PRINT_PAGE = "paper-print.html"
PDF_NAME = "paper.pdf"
TIMEOUT_S = 120


def chrome() -> str | None:
    """The Chrome to print with: ``$OPENINTERPRETABILITY_CHROME``, else a usual path, else None."""
    if named := os.environ.get(CHROME_ENV, "").strip():
        return named if Path(named).exists() or shutil.which(named) else None
    for path in CHROME_PATHS:
        if Path(path).exists():
            return path
    return next((found for name in CHROME_NAMES if (found := shutil.which(name))), None)


def print_pdf(page: Path, pdf: Path, browser: str) -> bool:
    """Print ``page`` to ``pdf`` with headless Chrome. True when the PDF was written."""
    args = [
        browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--force-prefers-reduced-motion", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf.resolve()}", page.resolve().as_uri(),
    ]  # fmt: skip
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        args.insert(1, "--no-sandbox")  # Chrome refuses to sandbox as root, as in a container
    try:
        subprocess.run(args, capture_output=True, timeout=TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"


@cache
def font_faces(math: bool = True) -> str:
    """The ``@font-face`` rules of the print page, each font a ``data:`` URI.

    The print page then depends on no system font, and Chrome needs no file access from it.
    ``math`` adds :data:`MATH_FACE`.
    """
    rules = []
    for family, weight, style, name in (*FONT_FACES, *([MATH_FACE] if math else [])):
        kind = "woff2" if name.endswith(".woff2") else "opentype"
        mime = "font/woff2" if kind == "woff2" else "font/otf"
        data = base64.b64encode((FONTS / name).read_bytes()).decode("ascii")
        rules.append(f'@font-face {{ font-family: "{family}"; font-weight: {weight}; '
                     f"font-style: {style}; src: url(data:{mime};base64,{data}) "
                     f'format("{kind}"); }}')  # fmt: skip
    return "\n".join(rules)


def _plain(raised: str) -> str:
    return raised.translate(str.maketrans(_SUPER, _PLAIN))


def raise_superscripts(text: str, svg: bool = False) -> str:
    """``text`` with each run of superscript characters as raised plain ones.

    HTML gets ``<sup>``. An SVG figure gets a raised, smaller ``<tspan>``. Only text
    between tags changes, so an attribute keeps its characters.
    """
    tag = '<tspan baseline-shift="super" font-size="70%">{}</tspan>' if svg else "<sup>{}</sup>"

    def text_node(node: re.Match) -> str:  # never an attribute, where a tag would break it
        return _SUPERS.sub(lambda m: tag.format(_plain(m.group(0))), node.group(0))

    return re.sub(r">[^<>]*<", text_node, text)


_FIGURE = re.compile(r'(<img\b[^>]*\bsrc=")(media/[^"]+\.svg)(")')
_SVG_OPEN = re.compile(r"<svg\b[^>]*>")
# The figures' own fonts are system faces. In the PDF, their text is set in CMU too.
FIGURE_FONTS = (
    'text, tspan { font-family: "CMU Serif", "Latin Modern Math", serif !important; }\n'
    '[font-family*="monospace"], [style*="monospace"] '
    '{ font-family: "CMU Typewriter Text", "Latin Modern Math", monospace !important; }'
)


def print_svg(svg: str) -> str:
    """``svg`` with the CMU fonts embedded and its text set in them, for the PDF.

    A figure is an image, and an SVG image loads no file. So the fonts go in as ``data:``
    URIs, in a style element right after the opening ``<svg>`` tag.
    """
    found = _SVG_OPEN.search(svg)
    if found is None:
        return svg
    style = f"<style>{font_faces(math=bool(_MATH.search(svg)))}\n{FIGURE_FONTS}</style>"
    return svg[: found.end()] + style + raise_superscripts(svg[found.end() :], svg=True)


def print_body(body: str, folder: Path) -> str:
    """The paper's HTML with each SVG figure as a ``data:`` URI of its :func:`print_svg`."""

    def inline(match: re.Match) -> str:
        path = folder / match.group(2)
        if not path.exists():
            return match.group(0)
        svg = print_svg(path.read_text(encoding="utf-8"))
        data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"{match.group(1)}data:image/svg+xml;base64,{data}{match.group(3)}"

    # U+FFFD, the replacement character of an undecodable token, is in no Computer Modern
    # face. The PDF names it instead, as code would.
    return raise_superscripts(_FIGURE.sub(inline, body)).replace("\ufffd", "\\ufffd")


def write_pdf(env, folder: Path, paper, common: dict, browser: str | None) -> bool:
    """Write ``paper.pdf`` in ``folder``, the paper's built folder. True when it was written.

    ``env`` is the site's Jinja environment and ``common`` its site-wide context.
    """
    if browser is None:
        return False
    page = folder / PRINT_PAGE
    template = env.get_template("paper_print.html")
    body = print_body(paper.body, folder)
    page.write_text(template.render(**common, paper=paper, body=body, font_faces=font_faces()))
    try:
        return print_pdf(page, folder / PDF_NAME, browser)
    finally:
        page.unlink(missing_ok=True)
