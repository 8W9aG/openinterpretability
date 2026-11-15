"""The site's colours for statuses and poles, in one place (issue #3279).

The status colours stay apart with red-green and blue-yellow colour blindness, in light
and dark mode. ``tests/test_palette.py`` checks it. VERIFIED and CONTESTED are Okabe and
Ito's bluish green and blue. NEEDS_REVISION and REJECTED are a burnt orange and a dark
red. They are darker than Okabe and Ito's orange and vermillion, because those two
collide in dark mode. Under deuteranopia the closest pair is still ΔE 10.9 (OKLab × 100).

The pole colours are Tailwind's orange-600 and sky-600, already a safe pair.

Templates use them as Tailwind colours (``status-verified``, ``pole-positive``), which
``base.html`` adds to its config from :func:`tailwind_colors`. The token heatmaps use
:data:`POLE_RGB`.
"""

from __future__ import annotations

STATUS_HEX = {
    "VERIFIED": "#009E73",
    "CONTESTED": "#0072B2",
    "NEEDS_REVISION": "#C46E00",
    "REJECTED": "#B42828",
}
POLE_HEX = {"positive": "#EA580C", "negative": "#0284C7"}
# Light and dark page surfaces (base.html: bg-white, dark:bg-slate-950).
SURFACES = {"light": "#FFFFFF", "dark": "#020617"}


def token(status: str) -> str:
    """The Tailwind colour name of ``status``: ``status-needs-revision``."""
    return "status-" + status.lower().replace("_", "-")


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
