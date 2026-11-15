"""The site's colours for statuses and poles, in one place (issue #3279).

The maintainer has red-green colour blindness (deuteranopia). On 2026-11-15 they chose
this status palette from candidates. VERIFIED is blue, CONTESTED violet, NEEDS_REVISION
pale yellow and REJECTED orange. The four differ in lightness as well as hue, so they
stay apart for that reader. ``tests/test_palette.py`` checks the separation that matters,
and that badge text stays readable on each fill.

The pole colours are Tailwind's orange-600 and sky-600, already a safe pair.

Templates use them as Tailwind colours (``status-verified``, ``pole-positive``), which
``builder/css.py`` adds to Tailwind's config from :func:`tailwind_colors`. The token heatmaps use
:data:`POLE_RGB`.
"""

from __future__ import annotations

STATUS_HEX = {
    "VERIFIED": "#1D4ED8",
    "CONTESTED": "#8B5CF6",
    "NEEDS_REVISION": "#FDE047",
    "REJECTED": "#F97316",
}
POLE_HEX = {"positive": "#EA580C", "negative": "#0284C7"}
# Light and dark page surfaces (base.html: bg-white, dark:bg-slate-950).
SURFACES = {"light": "#FFFFFF", "dark": "#020617"}


def token(status: str) -> str:
    """The Tailwind colour name of ``status``: ``status-needs-revision``."""
    return "status-" + status.lower().replace("_", "-")


def luminance(hex_colour: str) -> float:
    """The WCAG relative luminance of a hex colour."""
    channels = (int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def badge_text(hex_colour: str) -> str:
    """The text colour for a badge filled with ``hex_colour``: near-black or white."""
    return "#111111" if luminance(hex_colour) > 0.25 else "#FFFFFF"


def rgb(hex_colour: str) -> str:
    """``"234 88 12"``: a hex colour as the space-separated triplet CSS ``rgb()`` takes."""
    h = hex_colour.lstrip("#")
    return " ".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))


POLE_RGB = {pole: rgb(hex_colour) for pole, hex_colour in POLE_HEX.items()}


def tailwind_colors() -> dict[str, str]:
    """The colours ``base.html`` adds to Tailwind's theme."""
    colours = {token(s): h for s, h in STATUS_HEX.items()}
    colours |= {f"pole-{p}": h for p, h in POLE_HEX.items()}
    return colours
