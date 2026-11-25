"""labinacan's renderer, redirect pages, owned paths and number filters (#6628)."""

from types import SimpleNamespace

from jinja2 import DictLoader, Environment

from labinacan.site import render as r


def test_root_for_climbs_one_folder_per_level():
    assert [r.root_for(p) for p in ("a.html", "m/a.html", "m/l/a.html")] == ["", "../", "../../"]


def test_a_page_gets_the_common_context_its_root_and_its_url(tmp_path):
    templates = DictLoader({"t.html": "{{ site }}|{{ root }}|{{ canonical }}|{{ x }}"})
    env = Environment(loader=templates, autoescape=True)
    report = SimpleNamespace(pages=[])
    render = r.Renderer(env, tmp_path, {"site": "S", "site_url": "https://x.org/"}, report)
    assert render.jobs == 1
    render("m/a.html", "t.html", x=7)
    out = tmp_path / "m" / "a.html"
    assert out.read_text() == "S|../|https://x.org/m/a.html|7"
    assert report.pages == [out]


def test_a_redirect_page_points_at_the_new_page_and_is_not_indexed(tmp_path):
    report = SimpleNamespace(pages=[])
    out = tmp_path / "a" / "b" / "old.html"
    r.write_redirect(out, "../new.html", report)
    assert out.read_text() == (
        '<!doctype html><meta charset="utf-8"><title>Moved</title>'
        '<link rel="canonical" href="../new.html"><meta name="robots" content="noindex">'
        '<meta http-equiv="refresh" content="0; url=../new.html">'
        '<p>This page moved to <a href="../new.html">../new.html</a>.</p>\n'
    )
    assert report.pages == [out]


def test_clean_owned_removes_owned_files_and_folders_only(tmp_path):
    (tmp_path / "models" / "m").mkdir(parents=True)
    (tmp_path / "models" / "m" / "a.html").write_text("x")
    (tmp_path / "index.html").write_text("x")
    (tmp_path / "keep.html").write_text("x")
    r.clean_owned(tmp_path, ["models", "index.html", "absent"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.html"]


def test_pct_fine_keeps_two_significant_figures_of_small_shares():
    assert r.pct_fine(None) == "—" and r.pct_fine(0) == "0%" and r.pct_fine(0.5) == "50%"
    assert r.pct_fine(0.021) == "2.1%" and r.pct_fine(1 / 116_736) == "0.00086%"
    assert r.pct_fine(-0.0009418) == "-0.094%"  # a random baseline below zero


def test_the_number_filters():
    env = Environment(autoescape=True)
    assert r.number_filters(env) is env
    f = env.filters
    assert (f["pct"](0.123), f["pct"](None)) == ("12%", "—")
    assert f["pct_fine"] is r.pct_fine
    assert (f["num"](0.12345), f["num"](0.12345, 1), f["num"](None)) == ("0.123", "0.1", "—")
    assert (f["thousands"](1234567), f["thousands"](None)) == ("1,234,567", "?")
    assert (f["pval"](0.001), f["pval"](0.0009), f["pval"](None)) == ("0.001", "9.0e-04", "—")
