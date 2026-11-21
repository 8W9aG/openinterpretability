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


def test_a_region_outline_is_convex_holds_the_core_and_leaves_out_a_stray():
    from openinterpretability.builder.cluster_map import hull, region_outline

    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0)]
    assert sorted(hull(square)) == sorted(square[:4])  # the centre point is inside
    core = [(x, y) for x in range(0, 20, 2) for y in range(0, 20, 2)]
    path = region_outline([(float(x), float(y)) for x, y in core] + [(500.0, 500.0)])
    xs = [float(v) for v in path.replace("M", "").replace("L", "").replace("Z", "").split()[::2]]
    assert path.startswith("M") and path.endswith("Z")
    assert max(xs) < 100  # the stray at 500 is outside the inner 85%
    assert region_outline([(0.0, 0.0), (1.0, 1.0)]) == ""  # too few points for a shape


def test_the_map_draws_outlines_and_a_card_line(tmp_path, record):
    from openinterpretability.builder.cluster_map import overlaps
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.tree import build_tree

    (model,) = build_tree([record], [])
    found = _shared(model, record.submission.identifier)
    found["points"] += [dict(found["points"][1], slug=f"x{i}", neuron=10 + i,
                             x=0.1 * i, y=0.9 - 0.05 * i) for i in range(4)]  # fmt: skip
    view = map_view(model, found)
    assert [rid for _, _, _, rid in view["regions"]] == [1]  # only region 1 has 3+ points
    boxes = [b.box for b in view["badges"]]
    assert not any(overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    own = next(p for p in view["points"] if p.own)
    explanation = record.submission.natural_language_explanation
    assert own.why and explanation.startswith(own.why.rstrip("…").rstrip())
    compile_site([record], tmp_path / "site", concept_map=found)
    page = (tmp_path / "site" / "models" / model.slug / "index.html").read_text()
    assert 'class="region f0"' in page and 'data-cluster="0"' in page


def _studied(record, when, explanation, text, model=None):
    """``record`` at time ``when``, with one top example of ``text``."""
    from datetime import UTC, datetime

    from tests.test_dead import PROVENANCE

    example = record.submission.top_examples[0]
    tokens = list(text)
    ident = record.submission.identifier.model_copy(
        update={"model_name": model or record.submission.identifier.model_name}
    )
    sub = record.submission.model_copy(update={
        "identifier": ident, "provenance": PROVENANCE, "natural_language_explanation": explanation,
        "top_examples": [example.model_copy(update={
            "text": text, "tokens": tokens, "activations": [1.0] * len(tokens),
            "max_token_index": 0})]})  # fmt: skip
    verdict = record.verdict.model_copy(update={"timestamp": datetime(*when, tzinfo=UTC)})
    return record.model_copy(update={"submission": sub, "verdict": verdict})


def test_studied_poles_take_the_newest_record_with_texts_and_the_first_date(
    tmp_path, record, monkeypatch
):
    from openinterpretability.builder.wiki import archive_record
    from openinterpretability.data import sample_corpus

    monkeypatch.setenv("OPENINTERPRETABILITY_OFFLOAD_EVIDENCE", "0")
    wiki, model = tmp_path / "wiki", record.submission.identifier.model_name
    sample = sample_corpus()[0]  # a sample-corpus line says nothing, so it is left out
    for r in [
        _studied(record, (2026, 9, 30), "oldest, no texts", sample),
        _studied(record, (2026, 10, 1), "has texts", "quote"),
        _studied(record, (2026, 10, 2), "newest, no texts", sample),
        _studied(record, (2026, 10, 3), "another model", "quote", model="org/other"),
    ]:
        archive_record(r, wiki)
    (pole,) = cm.studied_poles({model}, wiki)
    assert (pole.model, pole.explanation, pole.texts) == (model, "has texts", ("«q»uote",))
    assert pole.first == "2026-09-30" and pole.corpus == "corpus.jsonl"


def test_clusters_split_two_far_groups_at_the_target_count():
    rng = np.random.default_rng(0)
    near = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    rows = [c + 0.01 * rng.normal(size=3) for c in near for _ in range(6)]
    vectors = np.array([r / np.linalg.norm(r) for r in rows])
    member = cm.clusters(vectors, k=3, target=2)
    assert len(set(member)) == 2
    assert len(set(member[:6])) == 1 and len(set(member[6:])) == 1


def test_layout_scales_each_axis_to_the_unit_square(monkeypatch):
    import sys
    import types

    made = []

    class UMAP:  # UMAP itself takes seconds to compile, so a fixed layout stands in
        def __init__(self, **kw):
            made.append(kw)

        def fit_transform(self, vectors):
            return np.array([[2.0, 5.0], [4.0, 5.0], [3.0, 5.0]])

    module = types.ModuleType("umap")
    module.UMAP = UMAP
    monkeypatch.setitem(sys.modules, "umap", module)
    vectors = np.eye(3)
    assert cm.layout(vectors) == [(0.0, 0.0), (1.0, 0.0), (0.5, 0.0)]  # a flat axis stays 0
    assert made[0] == {"n_components": 2, "metric": "cosine", "random_state": 0,
                       "init": "spectral", "n_neighbors": 2}  # fmt: skip
    cm.layout(vectors, init=[[0, 1], [1, 0], [0.5, 0.5]])
    assert made[1]["init"].tolist() == [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]


def test_a_refit_with_no_old_pole_starts_from_scratch():
    old = cm.build([_pole(1), _pole(2), _pole(3)], _encode, place=_fit, group=_group)
    _fit.init = "not called"
    found = cm.build([_pole(4), _pole(5), _pole(6)], _encode, old, refit=True,
                     place=_fit, group=_group)  # fmt: skip
    assert _fit.init is None and found["version"] == 2
    assert [p["slug"] for p in found["points"]] == [_pole(i).slug for i in (4, 5, 6)]
