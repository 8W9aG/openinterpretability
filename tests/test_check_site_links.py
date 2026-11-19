"""The built site's internal links and anchors (scripts/check_site_links.py, #3170)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_site_links.py"
spec = importlib.util.spec_from_file_location("check_site_links", SCRIPT)
links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(links)


def _site(tmp_path: Path) -> Path:
    site = tmp_path / "dist"
    (site / "models" / "m" / "L0").mkdir(parents=True)
    (site / "about").mkdir()
    (site / "about" / "index.html").write_text('<h2 id="team">Team</h2>')
    (site / "models" / "m" / "L0" / "N1.html").write_text(
        '<a href="../../../index.html#top">home</a><a name="old"></a>'
        '<img src="../../../logo.svg"><a href="#old">here</a>'
    )
    (site / "logo.svg").write_text("<svg/>")
    (site / "index.html").write_text(
        '<div id="top"></div>'
        '<a href="models/m/L0/N1.html">a unit</a>'
        '<a href="about/">about</a><a href="about/#team">the team</a>'
        '<a href="https://arxiv.org/abs/1">external</a><a href="mailto:x@y.z">mail</a>'
        '<a href="/models/m/L0/N1.html?x=1">from the root</a>'
    )
    return site


def test_a_site_whose_links_all_resolve_passes(tmp_path):
    site = _site(tmp_path)
    assert links.broken_links(site) == []
    assert links.main([str(site)]) == 0


def test_a_missing_page_a_missing_anchor_and_an_escape_are_found(tmp_path):
    site = _site(tmp_path)
    (site / "broken.html").write_text(
        '<a href="models/m/L0/N2.html">no page</a>'
        '<a href="about/#nobody">no anchor</a>'
        '<a href="#nowhere">no anchor here</a>'
        '<a href="../outside.html">outside</a>'
    )
    found = {(link, why) for page, link, why in links.broken_links(site) if page == "broken.html"}
    assert found == {
        ("models/m/L0/N2.html", "no such file"),
        ("about/#nobody", "no #nobody in it"),
        ("#nowhere", "no #nowhere in it"),
        ("../outside.html", "leaves the site"),
    }
    assert links.main([str(site)]) == 1
