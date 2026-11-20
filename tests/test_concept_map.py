"""The shared concept map of every model's studied neurons (concept_map.py, #4422, #4586)."""

import numpy as np
import pytest

from openinterpretability import concept_map as cm
from openinterpretability.builder.concept_map_view import HEIGHT, PAD, WIDTH, atlas_view, map_view
from openinterpretability.concept_regions import coverage, intersection, name_clusters

A, B = "org/model-a", "org/model-b"


def _pole(i, pole="positive", status="VERIFIED", explanation="fires on brackets in code",
          model=A, text=None):  # fmt: skip
    return cm.Pole(model, f"{model}/L3/N{i}-{pole}", 3, i, pole, (text or f"text {i}",),
                   explanation, status, f"2026-11-{10 + i:02d}")  # fmt: skip


def _encode(texts):
    """A fake embedder: the first word of each text, as a number, on two axes."""
    return np.array([[float(t.split()[-1]), 1.0] for t in texts])


def _fit(vectors, init=None):
    _fit.init = init
    return [(round(float(i) / max(len(vectors) - 1, 1), 4), 0.5) for i in range(len(vectors))]


def _group(vectors):
    return [0 if i % 2 == 0 else 1 for i in range(len(vectors))]


def test_pole_vectors_are_centred_and_normalised():
    raw = cm.means([_pole(0), _pole(1), _pole(2)], lambda t: [[float(len(t[0]))]])
    vectors, norms = cm.centred(raw, raw.mean(0))
    assert np.allclose(np.linalg.norm(vectors, axis=1)[norms > 0], 1.0)
    assert np.isclose(sum(vectors[:, 0] * norms), 0.0)  # centred on the mean of the poles


def test_the_first_fit_pools_every_model_on_one_centre():
    poles = [_pole(1), _pole(2), _pole(3, model=B), _pole(4, model=B)]
    found = cm.build(poles, _encode, place=_fit, group=_group)
    assert found["version"] == 1 and found["models"] == [A, B] and found["poles"] == 4
    assert found["centre"] == [2.5, 1.0]  # the mean over both models, not one per model
    assert [p["model"] for p in found["points"]] == [A, A, B, B]
    assert found["points"][0]["first"] == "2026-11-11"


def test_new_poles_move_no_old_point():
    old = cm.build([_pole(1), _pole(2), _pole(3, model=B)], _encode, place=_fit, group=_group)
    poles = [_pole(1), _pole(2), _pole(3, model=B), _pole(9, model=B, text="text 3")]
    found = cm.build(poles, _encode, old, place=lambda v: pytest.fail("no refit"), group=_group)
    by_slug = {p["slug"]: p for p in found["points"]}
    for p in old["points"]:  # each old pole is where it was, in its region
        assert (by_slug[p["slug"]]["x"], by_slug[p["slug"]]["cluster"]) == (p["x"], p["cluster"])
    assert found["version"] == 1 and found["centre"] == old["centre"]
    new, twin = by_slug[f"{B}/L3/N9-positive"], by_slug[f"{B}/L3/N3-positive"]
    assert (new["x"], new["y"], new["cluster"]) == (twin["x"], twin["y"], twin["cluster"])


def test_place_new_takes_the_weighted_mean_of_the_nearest():
    old = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    points = [{"x": 0.0, "y": 0.0, "cluster": 0}, {"x": 1.0, "y": 1.0, "cluster": 1},
              {"x": 9.0, "y": 9.0, "cluster": 2}]  # fmt: skip
    ((x, y, region),) = cm.place_new(old, points, np.array([[0.6, 0.8]]), k=2)
    assert (x, y) == (pytest.approx(0.5714, abs=1e-4), pytest.approx(0.5714, abs=1e-4))
    assert region == 1  # the heavier neighbour; the opposite pole weighs nothing


def test_a_refit_starts_from_the_old_positions_and_counts_up():
    old = cm.build([_pole(1), _pole(2), _pole(3)], _encode, place=_fit, group=_group)
    found = cm.build([_pole(1), _pole(2), _pole(3), _pole(4)], _encode, old, refit=True,
                     place=_fit, group=_group)  # fmt: skip
    assert found["version"] == 2
    assert [tuple(r) for r in _fit.init[:3]] == [(p["x"], p["y"]) for p in old["points"]]


def test_regions_are_named_from_verified_members_of_every_model():
    poles = [_pole(0, explanation="opening round brackets in code"),
             _pole(1, explanation="round brackets in python code", model=B),
             _pole(2, status="REJECTED", explanation="french words in prose"),
             _pole(3, explanation="french words in prose and poetry", model=B),
             _pole(4, explanation="french prose words"),
             _pole(5, status="NEEDS_REVISION", explanation="dates")]  # fmt: skip
    by_id = {c["id"]: c for c in name_clusters(poles, [0, 0, 0, 1, 1, 2])}
    assert by_id[0]["verified"] and "brackets" in by_id[0]["name"]
    assert by_id[1]["verified"] and "french" in by_id[1]["name"]
    assert by_id[2] == {"id": 2, "size": 1, "name": None, "verified": False}


def test_coverage_counts_each_model_in_each_region():
    points = [{"model": A, "cluster": 0}, {"model": A, "cluster": 1}, {"model": B, "cluster": 0},
              {"model": B, "cluster": 0}]  # fmt: skip
    found = coverage(points, [0, 1, 2])
    assert found["models"][A] == {"poles": 2, "regions": {"0": 1, "1": 1}, "coverage": 0.6667}
    assert found["models"][B]["coverage"] == 0.3333
    (pair,) = found["pairs"]
    assert (pair["a"], pair["b"], pair["overlap"]) == (A, B, 0.5)
    assert 0 <= pair["null"] <= 1
    from collections import Counter

    assert intersection(Counter({0: 3}), Counter({0: 1})) == 1.0  # alike spreads match fully


def test_the_file_keeps_one_point_to_a_line(tmp_path):
    found = cm.build([_pole(1), _pole(2), _pole(3)], _encode, place=_fit, group=_group)
    path = tmp_path / "shared.json"
    cm.write(found, path)
    assert cm.load(path) == found
    assert sum('"slug"' in line for line in path.read_text().splitlines()) == 3
    assert cm.load(tmp_path / "missing.json") is None


def _shared(model, ident):
    point = {"model": model.name, "slug": ident.slug, "layer": ident.layer,
             "neuron": ident.neuron_idx, "pole": ident.pole, "x": 1.0, "y": 0.0, "cluster": 0,
             "norm": 1.0, "first": "2026-11-02"}  # fmt: skip
    other = {"model": B, "slug": f"{B}/L9/N9-negative", "layer": 9, "neuron": 9,
             "pole": "negative", "x": 0.0, "y": 1.0, "cluster": 1, "norm": 1.0,
             "first": "2026-11-01"}  # fmt: skip
    clusters = [{"id": 0, "size": 1, "name": "quotes · speech", "verified": True},
                {"id": 1, "size": 1, "name": None, "verified": False}]  # fmt: skip
    reach = coverage([point, other], [0, 1])
    return {"version": 1, "fitted": "2026-11-20", "date": "2026-11-20", "embedder": cm.EMBEDDER,
            "models": sorted([model.name, B]), "poles": 2, "clusters": clusters,
            "coverage": reach, "points": [point, other], "centre": [0.0]}  # fmt: skip


def test_the_model_page_marks_its_own_poles(tmp_path, record):
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.tree import build_tree

    (model,) = build_tree([record], [])
    found = _shared(model, record.submission.identifier)
    view = map_view(model, found)
    other, own = view["points"]  # the other models' poles are drawn first, beneath
    assert own.own and not other.own and view["own"] == 1
    assert own.shape == "circle" and (own.x, own.y) == (WIDTH - PAD, HEIGHT - PAD)
    assert other.href.startswith("models/org--model-b/")
    assert [(b.number, b.slot) for b in view["badges"]] == [(1, 1), (2, 0)]
    assert view["counts"] == {1: 1}
    compile_site([record], tmp_path / "site", concept_map=found)
    page = (tmp_path / "site" / "models" / model.slug / "index.html").read_text()
    assert 'id="concept-map"' in page and "quotes · speech" in page and "(1 of 1)" in page
    assert "hollow rings" in page and "models/atlas.html" in page
    atlas = (tmp_path / "site" / "models" / "atlas.html").read_text()
    assert "Concept atlas" in atlas and "Shuffled null" in atlas and "animation-delay" in atlas
    assert "models/atlas.html" in (tmp_path / "site" / "models" / "index.html").read_text()
    compile_site([record], tmp_path / "bare")
    bare = (tmp_path / "bare" / "models" / model.slug / "index.html").read_text()
    assert 'id="concept-map"' not in bare
    assert not (tmp_path / "bare" / "models" / "atlas.html").exists()


def test_the_atlas_fills_in_by_first_study(record):
    from openinterpretability.builder.tree import build_tree

    (model,) = build_tree([record], [])
    view = atlas_view([model], _shared(model, record.submission.identifier))
    assert [p.model for p in view["points"]] == [B, model.name]  # B's pole was studied first
    assert [r["shape"] for r in view["rows"]] == ["circle", "square"]
    assert view["rows"][0]["by_number"] or view["rows"][1]["by_number"]
