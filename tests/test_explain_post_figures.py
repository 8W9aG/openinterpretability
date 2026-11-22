"""The explain post's figures match their records (scripts/figures_explain_post.py, #5851)."""

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from figures_explain_post import MEDIA, build  # noqa: E402

BUILT = build()


@pytest.mark.parametrize("name", sorted(BUILT))
def test_committed_figure_matches_its_data(name):
    assert (MEDIA / name).read_text(encoding="utf-8") == BUILT[name]


@pytest.mark.parametrize("name", sorted(BUILT))
def test_figure_is_valid_xml_with_a_text_alternative(name):
    title = ElementTree.fromstring(BUILT[name]).find("{http://www.w3.org/2000/svg}title")
    assert title is not None and title.text
