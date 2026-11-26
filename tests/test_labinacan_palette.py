"""labinacan's colour maths, against values worked out by hand from WCAG 2.x (#6628)."""

import pytest

from labinacan.site.palette import badge_text, luminance, rgb


def test_each_channel_has_its_wcag_weight():
    assert luminance("#FF0000") == pytest.approx(0.2126)
    assert luminance("#00FF00") == pytest.approx(0.7152)
    assert luminance("#0000FF") == pytest.approx(0.0722)
    assert luminance("FFFFFF") == pytest.approx(1.0)  # the "#" is optional


def test_dark_and_light_channels_take_their_own_curves():
    dark = 10 / 255 / 12.92  # at or below 0.04045, a channel is linear
    assert luminance("#0A0000") == pytest.approx(0.2126 * dark)
    light = ((128 / 255 + 0.055) / 1.055) ** 2.4  # above it, a power curve
    assert luminance("#800000") == pytest.approx(0.2126 * light)


def test_badge_text_is_dark_on_light_fills_and_white_on_dark_ones():
    assert badge_text("#FACC15") == "#111111"
    assert badge_text("#1D4ED8") == "#FFFFFF"


def test_rgb_is_the_three_channels_in_decimal():
    assert rgb("#EA580C") == "234 88 12"
    assert rgb("EA580C") == "234 88 12"
    assert rgb("#0A0B0C") == "10 11 12"
