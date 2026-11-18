"""The site's colours for statuses and poles, in one place (issue #3279).

The maintainer has red-green colour blindness (deuteranopia). On 2026-11-15 they chose
this status palette from candidates. VERIFIED is blue, CONTESTED violet, NEEDS_REVISION
pale yellow and REJECTED orange. The four differ in lightness as well as hue, so they
stay apart for that reader. ``tests/test_palette.py`` checks the separation that matters,
and that badge text stays readable on each fill.

The pole colours are Tailwind's orange-600 and sky-600, already a safe pair.

Templates use them as Tailwind colours (``status-verified``, ``pole-positive``,
``brand-700``), which
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
# The brand's accent: links, highlights and the mark (#3289). These are Tailwind's emerald
# shades, under one name, so that changing the accent is one edit here.
BRAND_HEX = {
    50: "#ecfdf5", 100: "#d1fae5", 200: "#a7f3d0", 300: "#6ee7b7", 400: "#34d399",
    500: "#10b981", 600: "#059669", 700: "#047857", 800: "#065f46", 900: "#064e3b",
    950: "#022c22",
}  # fmt: skip
# The mark's colours: the model layouts' own yellow, orange and blue (#4465). VERIFIED's blue
# is too dark for the dark tile, so the blue is the negative pole's sky. They differ in
# lightness as well as hue, and none is a green.
MARK_HEX = {
    "yellow": STATUS_HEX["NEEDS_REVISION"],
    "orange": STATUS_HEX["REJECTED"],
    "blue": POLE_HEX["negative"],
}
MARK_TILE = "#0F172A"  # Tailwind's slate-900: the mark's tile
INK = {"light": "#0F172A", "dark": "#F8FAFC"}  # the wordmark: slate-900, and slate-50 in dark mode
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
    colours |= {f"brand-{shade}": h for shade, h in BRAND_HEX.items()}
    return colours
