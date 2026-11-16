"""A PDF of each paper, laid out like a conference paper (issue #3140).

The build writes a print page for the paper (``templates/paper_print.html``), with a
stylesheet in the style of a NeurIPS paper. Headless Chrome prints it to ``paper.pdf``
beside the paper's page, and the print page is then deleted. Chrome reads the figures'
CSS (their colour variables and the transforms of moving marks) as the site does. With
reduced motion forced on, each animated figure prints its resting frame.

A machine with no Chrome builds no PDFs, and the paper pages then show no PDF link.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CHROME_ENV = "OPENINTERP_CHROME"
# Where Chrome or Chromium usually is: macOS, then Linux (GitHub's runners have google-chrome).
CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)
CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
PRINT_PAGE = "paper-print.html"
PDF_NAME = "paper.pdf"
TIMEOUT_S = 120


def chrome() -> str | None:
    """The Chrome to print with: ``$OPENINTERP_CHROME``, else a usual path; None if none."""
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


def write_pdf(env, folder: Path, paper, common: dict, browser: str | None) -> bool:
    """Write ``paper.pdf`` in ``folder``, the paper's built folder. True when it was written.

    ``env`` is the site's Jinja environment and ``common`` its site-wide context.
    """
    if browser is None:
        return False
    page = folder / PRINT_PAGE
    page.write_text(env.get_template("paper_print.html").render(**common, paper=paper))
    try:
        return print_pdf(page, folder / PDF_NAME, browser)
    finally:
        page.unlink(missing_ok=True)
