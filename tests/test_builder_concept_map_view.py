"""The shared concept map as a model page and the atlas draw it (builder/concept_map_view.py)."""

from openinterpretability.builder.concept_map_view import atlas_view, map_view
from tests.test_concept_map import B, _shared


def test_a_model_with_no_pole_on_the_map_gets_no_map(record):
    from openinterpretability.builder.tree import build_tree

    (model,) = build_tree([record], [])
    found = _shared(model, record.submission.identifier)
    found["points"] = [p for p in found["points"] if p["model"] == B]  # only the other model's
    assert map_view(model, found) is None


def test_a_region_with_no_points_gets_no_badge(record):
    from openinterpretability.builder.tree import build_tree

    (model,) = build_tree([record], [])
    found = _shared(model, record.submission.identifier)
    found["clusters"].append({"id": 2, "size": 0, "name": None, "verified": False})
    view = atlas_view([model], found)
    assert [b.number for b in view["badges"]] == [1, 2]  # region 3 has no points
