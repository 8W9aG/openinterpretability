"""The concept map of a model's studied neurons (concept_map.py, concept_map_view.py, #4422)."""

import numpy as np
import pytest

from openinterpretability import concept_map as cm
from openinterpretability.builder.concept_map_view import HEIGHT, PAD, WIDTH, map_view

MODEL = "Qwen/Qwen2.5-0.5B"


def _pole(i, pole="positive", status="VERIFIED", explanation="fires on brackets in code"):
    return cm.Pole(f"m/L3/N{i}-{pole}", 3, i, pole, (f"text {i}",), explanation, status)


def _encode(texts):
    """A fake embedder: one axis per distinct word."""
    vocab = sorted({w for t in texts for w in t.split()})
    return np.array([[float(w in t.split()) for w in vocab] or [1.0] for t in texts])


def test_pole_vectors_are_centred_and_normalised():
    vectors, norms = cm.pole_vectors([_pole(0), _pole(1), _pole(2)], lambda t: [[float(len(t[0]))]])
    assert np.allclose(np.linalg.norm(vectors, axis=1)[norms > 0], 1.0)
    assert np.isclose(sum(vectors[:, 0] * norms), 0.0)  # centred on the mean of the poles


def test_clusters_are_named_from_verified_members_only():
    poles = [_pole(0, explanation="opening round brackets in code"),
             _pole(1, explanation="round brackets in python code"),
             _pole(2, status="REJECTED", explanation="french words in prose"),
             _pole(3, explanation="french words in prose and poetry"),
             _pole(4, explanation="french prose words"),
             _pole(5, status="NEEDS_REVISION", explanation="dates")]  # fmt: skip
    named = cm.name_clusters(poles, [0, 0, 0, 1, 1, 2])
    by_id = {c["id"]: c for c in named}
    assert by_id[0]["verified"] and "brackets" in by_id[0]["name"]
    assert "french" not in by_id[0]["name"]
    assert by_id[1]["verified"] and "french" in by_id[1]["name"]
    assert by_id[2] == {"id": 2, "size": 1, "name": None, "verified": False}  # no VERIFIED member
    assert [c["size"] for c in named] == [3, 2, 1]


def test_build_writes_points_clusters_and_provenance(tmp_path):
    poles = [_pole(i) for i in range(4)] + [_pole(0, "negative", "REJECTED")]
    found = cm.build(MODEL, poles, _encode, place=lambda v: [(0.1 * i, 0.5) for i in range(len(v))],
                     group=lambda v: [0, 0, 1, 1, 1])  # fmt: skip
    assert found["model"] == MODEL and found["embedder"] == cm.EMBEDDER and found["poles"] == 5
    assert found["points"][4] == {"slug": "m/L3/N0-negative", "layer": 3, "neuron": 0,
                                  "pole": "negative", "x": 0.4, "y": 0.5, "cluster": 1,
                                  "norm": pytest.approx(found["points"][4]["norm"])}  # fmt: skip
    path = cm.map_path(MODEL, tmp_path)
    cm.write(found, path)
    assert cm.load(path) == found and path.name == "Qwen--Qwen2.5-0.5B.json"
    assert cm.load(tmp_path / "missing.json") is None


def test_the_model_page_draws_the_map(tmp_path, record):
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.tree import build_tree

    (model,) = build_tree([record], [])
    ident = record.submission.identifier
    points = [{"slug": ident.slug, "layer": ident.layer, "neuron": ident.neuron_idx,
               "pole": ident.pole, "x": 1.0, "y": 0.0, "cluster": 0, "norm": 1.0},
              {"slug": "x", "layer": 9, "neuron": 9, "pole": "negative", "x": 0.0, "y": 1.0,
               "cluster": 1, "norm": 1.0}]  # fmt: skip
    clusters = [{"id": 0, "size": 1, "name": "quotes · speech", "verified": True},
                {"id": 1, "size": 1, "name": None, "verified": False}]  # fmt: skip
    found = {"model": model.name, "embedder": cm.EMBEDDER, "date": "2026-11-17", "poles": 2,
             "clusters": clusters, "points": points}  # fmt: skip
    view = map_view(model, found)
    studied, other = view["points"]
    assert studied.shape == "circle"
    assert studied.href.endswith(f"L{ident.layer}/N{ident.neuron_idx}.html")
    assert other.shape == "dot" and (studied.x, studied.y) == (WIDTH - PAD, HEIGHT - PAD)
    badges = view["badges"]  # numbered by size; only a cluster with a VERIFIED member is named
    assert [(b.number, b.name, b.slot) for b in badges] == [
        (1, "quotes · speech", 1),
        (2, "no VERIFIED member, so no name", 0),
    ]
    assert (studied.slot, other.slot) == (1, 0)  # a nameless cluster is grey
    assert all(b.box[0] >= 0 and b.box[2] <= WIDTH for b in badges)
    compile_site([record], tmp_path / "site", concept_maps={model.name: found})
    page = (tmp_path / "site" / "models" / model.slug / "index.html").read_text()
    assert 'id="concept-map"' in page and "quotes · speech" in page
    assert 'class="litmap' in page and "map-card" in page  # the shared theme (#4577)
    assert "Distances between clusters mean little" in page
    compile_site([record], tmp_path / "bare")
    bare = (tmp_path / "bare" / "models" / model.slug / "index.html").read_text()
    assert 'id="concept-map"' not in bare
