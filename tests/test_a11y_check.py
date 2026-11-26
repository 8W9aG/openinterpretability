"""Accessibility check (scripts/a11y_check.py, issue #3280): page choice and failure rule."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "a11y_check", Path(__file__).parents[1] / "scripts" / "a11y_check.py"
)
a11y = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(a11y)


@pytest.fixture(autouse=True)
def model_pages(monkeypatch):
    """A project whose own kinds of page are models, layers, units and experiments."""
    kinds = ("models/*/index.html", "models/*/L*/index.html", "models/*/L*/N*.html",
             "models/*/L*/N*-*/*.html")  # fmt: skip
    monkeypatch.setattr(a11y, "a11y_pages", lambda root: kinds)


def touch(root, *paths):
    for p in paths:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text("<html></html>")


def test_sample_pages_takes_home_and_the_first_of_each_kind(tmp_path):
    touch(
        tmp_path,
        "index.html",
        "models/b/index.html",
        "models/a/index.html",
        "models/a/L2/index.html",
        "models/a/L2/N7.html",
        "models/a/L2/N7-positive/20261019T000000Z.html",
        "models/a/L2/N7-positive/20261018T000000Z.html",
    )
    assert a11y.sample_pages(tmp_path) == [
        "index.html",
        "models/a/index.html",
        "models/a/L2/index.html",
        "models/a/L2/N7.html",
        "models/a/L2/N7-positive/20261018T000000Z.html",
    ]


def test_sample_pages_leaves_out_a_missing_kind(tmp_path):
    touch(tmp_path, "index.html", "models/a/index.html")
    assert a11y.sample_pages(tmp_path) == ["index.html", "models/a/index.html"]


def test_failures_counts_serious_and_critical_nodes_only():
    violations = [
        {
            "id": "color-contrast",
            "impact": "serious",
            "nodes": [{"target": ["p.a"]}, {"target": ["p.b"]}],
        },
        {"id": "label", "impact": "critical", "nodes": [{"target": ["#sql"]}]},
        {"id": "region", "impact": "moderate", "nodes": [{"target": ["div"]}]},
        {"id": "x", "impact": None, "nodes": [{"target": ["span"]}]},
    ]
    assert a11y.failures(violations) == [
        "color-contrast (serious): p.a",
        "color-contrast (serious): p.b",
        "label (critical): #sql",
    ]


def test_axe_source_refuses_a_file_with_the_wrong_hash(tmp_path):
    (tmp_path / f"axe-{a11y.AXE_VERSION}.min.js").write_text("not axe")
    try:
        a11y.axe_source(tmp_path)
    except SystemExit as e:
        assert "SHA-256" in str(e)
    else:
        raise AssertionError("a file with the wrong hash was accepted")
    assert not (tmp_path / f"axe-{a11y.AXE_VERSION}.min.js").exists()


def test_a_site_must_have_a_page_of_each_kind(tmp_path):
    # #6631: the home page, Research and each project kind, checked against the site itself.
    touch(tmp_path, "research/index.html", "models/a/index.html", "models/a/L2/index.html")
    assert a11y.missing_kinds(tmp_path) == [
        "index.html", "models/*/L*/N*.html", "models/*/L*/N*-*/*.html"
    ]  # fmt: skip
    touch(tmp_path, "index.html", "models/a/L2/N7.html", "models/a/L2/N7-positive/e.html")
    assert a11y.missing_kinds(tmp_path) == []
