"""The status colours stay apart with colour blindness, in light and dark (issue #3279).

Each pair of status colours is compared in OKLab (ΔE × 100) with normal vision and
under simulated protanopia, deuteranopia and tritanopia. The simulation is Machado,
Oliveira and Fernandes (2009) at full severity, applied in linear sRGB. Each colour must
also reach 3:1 contrast against both page surfaces, since it fills marks on them.
"""

import itertools
import math

import pytest

from openinterp.builder import palette

CVD = {
    "protanopia": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
                   (-0.003882, -0.048116, 1.051998)),
    "deuteranopia": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
                     (-0.011820, 0.042940, 0.968881)),
    "tritanopia": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602),
                   (0.004733, 0.691367, 0.303900)),
}  # fmt: skip
MIN_CVD, MIN_NORMAL, MIN_CONTRAST = 8.0, 15.0, 3.0


def linear(hex_colour: str) -> tuple[float, ...]:
    channels = (int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)


def oklab(rgb: tuple[float, ...]) -> tuple[float, ...]:
    r, g, b = (min(max(c, 0.0), 1.0) for c in rgb)
    lms = (
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    )
    l_, m_, s_ = (math.copysign(abs(x) ** (1 / 3), x) for x in lms)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def simulate(rgb: tuple[float, ...], matrix) -> tuple[float, ...]:
    return tuple(sum(row[i] * rgb[i] for i in range(3)) for row in matrix)


def delta_e(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return 100 * math.dist(oklab(a), oklab(b))


def luminance(hex_colour: str) -> float:
    r, g, b = linear(hex_colour)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


PAIRS = list(itertools.combinations(palette.STATUS_HEX.items(), 2))


@pytest.mark.parametrize(("a", "b"), PAIRS, ids=[f"{a[0]}-{b[0]}" for a, b in PAIRS])
def test_every_pair_of_statuses_stays_apart(a, b):
    ra, rb = linear(a[1]), linear(b[1])
    assert delta_e(ra, rb) >= MIN_NORMAL
    for name, matrix in CVD.items():
        assert delta_e(simulate(ra, matrix), simulate(rb, matrix)) >= MIN_CVD, name


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_status_stands_out_from_the_page(mode):
    surface = luminance(palette.SURFACES[mode])
    for status, hex_colour in palette.STATUS_HEX.items():
        hi, lo = sorted((luminance(hex_colour), surface), reverse=True)
        assert (hi + 0.05) / (lo + 0.05) >= MIN_CONTRAST, status


def test_the_old_green_and_red_would_fail():
    # emerald-500 and rose-500, the colours before #3279: apart for normal vision only.
    old = linear("#10B981"), linear("#F43F5E")
    assert (
        delta_e(simulate(old[0], CVD["deuteranopia"]), simulate(old[1], CVD["deuteranopia"]))
        < MIN_CVD
    )


def test_the_palette_feeds_tailwind_and_the_heatmaps():
    colours = palette.tailwind_colors()
    assert colours["status-needs-revision"] == palette.STATUS_HEX["NEEDS_REVISION"]
    assert palette.POLE_RGB["positive"] == "234 88 12"  # orange-600, unchanged
