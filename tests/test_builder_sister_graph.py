"""Edge selection in the cross-model sister graph."""

from openinterpretability.builder.sister_graph import sister_graphs
from openinterpretability.sisters import SisterRow


def _row(slug, other, matches, rotated=0.1):
    keys = ("layer", "neuron_idx", "pole", "pearson", "back_slug", "back_pearson")
    return SisterRow.model_validate(
        {
            "slug": slug,
            "other": other,
            "model_revision": None,
            "other_revision": None,
            "corpus": "c",
            "corpus_sha256": "0",
            "tokens": 10,
            "coverage": 1.0,
            "matches": [dict(zip(keys, m, strict=True)) for m in matches],
            "rotated": rotated,
        }
    )


def test_a_row_without_matches_draws_no_edge():
    rows = {"org--a/L1/N1-positive": [_row("org--a/L1/N1-positive", "org/b", [])]}
    model = type("M", (), {"name": "org/a", "n_layers": 2})()
    assert sister_graphs(model, rows, studied=set()) == []


def test_a_weaker_match_for_the_same_pair_keeps_the_stronger_edge():
    a, b = "org/a", "org/b"
    slug = "org--a/L1/N1-positive"
    rows = {
        slug: [
            _row(slug, b, [(2, 3, "positive", 0.7, "org--b/L0/N0-positive", 0.7)]),
            _row(slug, b, [(2, 3, "positive", 0.4, "org--b/L0/N0-positive", 0.4)]),
        ]
    }
    model = type("M", (), {"name": a, "n_layers": 2})()
    (g,) = sister_graphs(model, rows, studied=set())
    assert g["n_edges"] == 1 and g["n_sisters"] == 0
    assert g["lines"][0]["label"].endswith("r 0.70")
