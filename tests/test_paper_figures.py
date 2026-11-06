"""The media of the papers in papers/."""

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from figure_svg import gelu, number  # noqa: E402
from paper_figures import PAPERS, build_all, generated  # noqa: E402

BUILT = build_all()


@pytest.mark.parametrize("name", sorted(BUILT))
def test_committed_figure_matches_its_data(name):
    assert (PAPERS / name).read_text(encoding="utf-8") == BUILT[name]


@pytest.mark.parametrize("name", sorted(BUILT))
def test_figure_is_valid_xml_with_a_text_alternative(name):
    root = ElementTree.fromstring(BUILT[name])
    title = root.find("{http://www.w3.org/2000/svg}title")
    assert title is not None and title.text


@pytest.mark.parametrize("name", sorted(BUILT))
def test_each_figure_appears_in_its_paper(name):
    paper, _, file_name = name.split("/")
    prose = (PAPERS / paper / "README.md").read_text(encoding="utf-8")
    assert f"(media/{file_name})" in prose


def test_interactive_page_data_matches_the_results():
    written = (PAPERS / "below-the-dip" / "data.js").read_text(encoding="utf-8")
    assert written == generated()["below-the-dip/data.js"]


def test_each_media_file_is_built():
    on_disk = {str(path.relative_to(PAPERS)) for path in PAPERS.glob("*/media/*")}
    assert on_disk == set(BUILT)


def test_number_uses_a_true_minus_sign_and_no_negative_zero():
    assert number(-1.04) == "−1.04"
    assert number(-0.001) == "0.00"


def test_gelu_has_its_minimum_at_the_dip():
    dip = -0.7517915246935645
    assert gelu(dip) < gelu(dip - 0.01)
    assert gelu(dip) < gelu(dip + 0.01)
