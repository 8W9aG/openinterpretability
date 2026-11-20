"""The media of the papers in papers/."""

import re
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from figure_svg import gelu, number  # noqa: E402
from paper_figures import PAPERS, build_all  # noqa: E402

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


def test_each_media_file_is_built():
    from openinterpretability.builder.social_media import IMAGE, MANIFEST, VIDEO

    drawn = {IMAGE, VIDEO, MANIFEST}  # scripts/social_images.py draws these, not this script
    on_disk = {str(path.relative_to(PAPERS)) for path in PAPERS.glob("*/media/*")
               if path.name not in drawn}  # fmt: skip
    assert on_disk == set(BUILT)


def test_number_uses_a_true_minus_sign_and_no_negative_zero():
    assert number(-1.04) == "−1.04"
    assert number(-0.001) == "0.00"


def test_gelu_has_its_minimum_at_the_dip():
    dip = -0.7517915246935645
    assert gelu(dip) < gelu(dip - 0.01)
    assert gelu(dip) < gelu(dip + 0.01)


# Motion (the paper-animation skill, #4210). Cycles stay short, and every animated mark
# rests on its final frame when motion is off.
ANIMATED = sorted(name for name, svg in BUILT.items() if "@keyframes" in svg)
CYCLE_SECONDS = (4, 15)
MAX_BYTES = 200_000
RULE = re.compile(r"\.([\w-]+)\{([^{}]*animation:[^{}]*)\}")
RUN = re.compile(r"animation:([\w-]+) (\d+(?:\.\d+)?)s [\w-]+ infinite(?: alternate)?;?")
# A final frame that is a mark's natural state needs no resting CSS: it shows by default.
NATURAL = ("opacity:1", "transform:scaleX(1)")


def _rest_ok(svg: str, keyframes: str, rest: str) -> bool:
    """True when the mark has a resting state, or its last keyframe is its natural state."""
    if rest.strip(";"):
        return True
    body = re.search(rf"@keyframes {re.escape(keyframes)}\{{(.*?)\}}\}}", svg).group(1)
    return body.rsplit("{", 1)[1] in NATURAL


def test_some_papers_animate():
    papers = {name.split("/")[0] for name in ANIMATED}
    assert {"neuron-development", "native-vs-learned", "explanations-across-models"} <= papers


@pytest.mark.parametrize("name", ANIMATED)
def test_an_animated_figure_stops_for_reduced_motion_and_loops_briefly(name):
    svg = BUILT[name]
    assert "@media (prefers-reduced-motion: reduce){*{animation:none!important}}" in svg
    assert len(svg.encode()) <= MAX_BYTES
    rules = RULE.findall(svg)
    assert rules, f"{name}: no animated class"
    for _, rule in rules:
        keyframes, seconds = RUN.search(rule).groups()
        assert CYCLE_SECONDS[0] <= float(seconds) <= CYCLE_SECONDS[1], f"{name}: {keyframes}"
        rest = RUN.sub("", rule)
        assert _rest_ok(svg, keyframes, rest), f"{name}: {keyframes} has no resting state"


def test_development_dots_move_only_through_measured_checkpoints():
    import json

    import figures_development as fd

    rows = [json.loads(line) for line in (PAPERS / "neuron-development" / "results.jsonl").open()]
    steps = sorted({r["step"] for r in rows if r["table"] == "loss"})
    svg = BUILT["neuron-development/media/development.svg"]
    loss = re.search(r"@keyframes lossdot\{(.*?)\}\.lossdot", svg).group(1)
    assert loss.count("transform") == len(steps) + 1  # one per checkpoint, and the hold
    assert "lossdot{animation:lossdot 9s step-end" in svg and fd.CYCLE == "9s"
    assert svg.count('class="small stepnow') == len(steps)
