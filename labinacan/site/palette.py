"""Colour maths for a site's palette: WCAG luminance, badge text and CSS triplets (#6628).

A project keeps its own colours. These functions work on any hex colour.
"""

from __future__ import annotations


def luminance(hex_colour: str) -> float:
    """The WCAG relative luminance of a hex colour.

    Args:
        hex_colour: The colour, such as ``#1D4ED8``.

    Returns:
        The luminance, from 0 for black to 1 for white.

    Examples:
        >>> luminance("#FFFFFF"), luminance("#000000")
        (1.0, 0.0)
    """
    channels = (int(value) / 255 for value in rgb(hex_colour).split())
    r, g, b = (
        c / 12.92
        if c <= 0.04045  # pragma: no mutate, equivalent: no 8-bit channel is 0.04045
        else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    )
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def badge_text(hex_colour: str) -> str:
    """The text colour for a badge filled with ``hex_colour``: near-black or white.

    Args:
        hex_colour: The badge's fill.

    Returns:
        ``#111111`` on a light fill, else ``#FFFFFF``.

    Examples:
        >>> badge_text("#FDE047"), badge_text("#1D4ED8")
        ('#111111', '#FFFFFF')
    """
    light = luminance(hex_colour) > 0.25  # pragma: no mutate, equivalent: no 8-bit colour is 0.25
    return "#111111" if light else "#FFFFFF"


def rgb(hex_colour: str) -> str:
    """``"234 88 12"``: a hex colour as the space-separated triplet CSS ``rgb()`` takes.

    Args:
        hex_colour: The colour.

    Returns:
        Its red, green and blue values, 0 to 255, separated by spaces.

    Examples:
        >>> rgb("#EA580C")
        '234 88 12'
    """
    h = hex_colour.lstrip("#")  # pragma: no mutate, equivalent: hex digits hold no "X"
    return " ".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))
