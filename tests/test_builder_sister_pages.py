"""The sister-match pages and each model's index of them (#6639)."""

from datetime import datetime
from types import SimpleNamespace as NS

from openinterpretability.builder.hrefs import sister_index_href, sister_match_href
from openinterpretability.builder.sister_pages import (
    PoleView,
    by_model,
    render_sister_pages,
    sister_matches,
)
from openinterpretability.sisters import SisterRow

A, B = "org--a/L1/N1-positive", "org--b/L0/N4-negative"
A_PAGE, B_PAGE = "models/org--a/L1/N1.html", "models/org--b/L0/N4.html"


def _row(slug, other, match, back, pearson=0.8, rotated=0.1):
    layer, neuron, pole = match
    return SisterRow.model_validate(
        {"slug": slug, "other": other, "model_revision": None, "other_revision": None,
         "corpus": "c", "corpus_sha256": "0", "tokens": 10, "coverage": 1.0,
         "matches": [{"layer": layer, "neuron_idx": neuron, "pole": pole, "pearson": pearson,
                      "back_slug": back, "back_pearson": pearson}],
         "rotated": rotated}
    )  # fmt: skip


def _both_ways():
    """One sister match, found from each side, plus a best match that isn't a sister."""
    return {
        A: [_row(A, "org/b", (0, 4, "negative"), A, pearson=0.8)],
        B: [_row(B, "org/a", (1, 1, "positive"), B, pearson=0.9)],
        "org--a/L1/N2-positive": [
            _row("org--a/L1/N2-positive", "org/b", (0, 5, "positive"), "org--a/L1/N9-positive")
        ],
    }


def _models():
    """A tree with ``org/a`` only: pole A is VERIFIED there, and ``org/b`` isn't audited."""
    sub = NS(identifier=NS(slug=A), natural_language_explanation="the word cat")
    standing = NS(status="VERIFIED", lead=NS(record=NS(submission=sub)))
    node = NS(is_neuron=True, href=A_PAGE, standings={"positive": standing})
    head = NS(is_neuron=False)
    return [NS(name="org/a", slug="org--a", layers=[NS(units=[head, node])])]


def _study(tested, source, status, day, href):
    sub = NS(identifier=NS(slug=tested), sister_of=f"{source}/20260101T000000Z")
    verdict = NS(status=status, timestamp=datetime(2026, 1, day))
    return NS(record=NS(submission=sub, verdict=verdict), href=href)


def test_a_match_found_from_both_sides_is_one_page_with_both_rows():
    (m,) = sister_matches(_both_ways(), [], _models(), studied={A_PAGE})
    assert (m.a.slug, m.b.slug, len(m.rows)) == (A, B, 2)
    assert m.href == sister_match_href(B, A) and m.pearson == 0.9 and m.rotated == 0.1
    assert m.studies == []  # the page says no sister study has run yet
    assert (m.a.href, m.a.status, m.a.explanation) == (A_PAGE, "VERIFIED", "the word cat")
    assert (m.b.href, m.b.status, m.b.explanation) == (None, None, None)  # no page on the site


def test_studies_from_either_side_attach_to_the_match_newest_first():
    studies = [
        _study(B, A, "REJECTED", 1, "e/old.html"),
        _study(A, B, "VERIFIED", 3, "e/new.html"),
        _study(A, "org--b/L9/N9-positive", "VERIFIED", 2, "e/other.html"),  # not this match
        NS(record=NS(submission=NS(sister_of=None))),  # not a sister study
    ]
    (m,) = sister_matches(_both_ways(), studies, _models(), studied=set())
    assert [(s.tested, s.source, s.status, s.href) for s in m.studies] == [
        (A, B, "VERIFIED", "e/new.html"),
        (B, A, "REJECTED", "e/old.html"),
    ]


def test_matches_sort_strongest_first_then_by_page():
    c = "org--a/L2/N3-positive"

    def with_c(pearson):
        return _both_ways() | {c: [_row(c, "org/b", (1, 2, "positive"), c, pearson=pearson)]}

    assert [m.a.slug for m in sister_matches(with_c(0.95), [], _models(), set())] == [c, A]
    assert [m.a.slug for m in sister_matches(with_c(0.9), [], _models(), set())] == [A, c]


def test_a_pole_view_names_its_model_and_label():
    assert (PoleView(A, None).model, PoleView(A, None).label) == ("org/a", "L1/N1+")
    assert PoleView(B, None).label == "L0/N4−"


def test_each_match_is_indexed_under_both_models():
    (m,) = sister_matches(_both_ways(), [], _models(), studied=set())
    assert by_model([m]) == {"org/a": {"org/b": [m]}, "org/b": {"org/a": [m]}}


def test_render_writes_each_match_and_only_audited_models_indexes():
    calls = []
    render_sister_pages(lambda *a, **k: calls.append((a, k)), _models(), _both_ways(), [], set())
    (match, index) = calls
    assert match[0] == (sister_match_href(A, B), "sister_match.html")
    assert match[1]["m"].a.slug == A
    assert index[0] == (sister_index_href("org--a"), "sisters.html")
    assert index[1]["model"].slug == "org--a" and list(index[1]["groups"]) == ["org/b"]


def test_no_sister_means_no_pages():
    calls = []
    rows = {k: v for k, v in _both_ways().items() if k not in (A, B)}
    render_sister_pages(lambda *a, **k: calls.append(a), _models(), rows, [], set())
    assert calls == []
