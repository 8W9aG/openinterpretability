"""A paper's figure captions open with the figure's finding, its SVG title (#6930).

``write-paper`` puts the finding of each figure in its SVG title, which
``scripts/paper_figures.py`` draws from the results. The caption after **Figure N.** repeats
that title as its first sentence, and then says what the marks are. ``scripts/papers.py
check`` runs :func:`problems` on each preprint.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

# An image of the paper's media, a blank line, then its caption on one line.
FIGURE = re.compile(
    r"!\[[^\]]*\]\((media/[^)\s]+\.svg)\)[ \t]*\n[ \t]*\n\*\*Figure (\d+)\.\*\* ([^\n]*)"
)
TITLE = re.compile(r'<text[^>]*class="title"[^>]*>([^<]*)</text>')


def title(svg: str) -> str:
    """The figure's title: the text of its title lines, joined into one sentence.

    Args:
        svg: The figure's SVG.

    Returns:
        The title, with a full stop unless it ends in one, a question mark or a "!".

    Examples:
        >>> title('<text class="title">A rises,</text><text class="title">and B &amp; C</text>')
        'A rises, and B & C.'
        >>> title('<text x="1" class="title">Does A rise?</text>')
        'Does A rise?'
    """
    text = " ".join(" ".join(html.unescape(t).split()) for t in TITLE.findall(svg)).strip()
    return text if text.endswith((".", "?", "!")) else f"{text}."


def problems(folder: Path) -> list[str]:
    """Each figure caption of the paper in ``folder`` that does not open with its title.

    Args:
        folder: The paper's folder, with ``README.md`` and ``media/``.

    Returns:
        One problem for each such caption, naming its figure and its title.
    """
    text = (folder / "README.md").read_text(encoding="utf-8")
    out = []
    for path, number, caption in FIGURE.findall(text):
        svg = folder / path
        if not svg.exists():
            continue  # a missing figure is the figures' check's problem
        want = title(svg.read_text(encoding="utf-8"))
        if want != "." and not " ".join(caption.split()).startswith(want):
            out.append(f"Figure {number}'s caption does not open with its finding: {want!r}")
    return out
