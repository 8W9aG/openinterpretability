"""A model's 3D embedding space, continuous through its layers (#4929)."""

import torch

from openinterpretability import embedding_space as es

TEXTS = ["a b c d e f g h", "h g f e d c b a", "a c e g b d f h"] * 40


def _runner(tiny_gpt2):
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    return NeuronHookRunner(tiny_gpt2.double(), FakeTokenizer())


def test_the_space_fits_and_projects_the_corpus_onto_its_own_cloud(
    tiny_gpt2, tmp_path, monkeypatch
):
    monkeypatch.setattr(es, "WINDOW", 8)
    runner = _runner(tiny_gpt2)
    states = es.corpus_states(runner, TEXTS, points=120)
    assert states.shape[:2] == (3, 120)  # the embedding and 2 layers
    space = es.fit(states)
    assert space["cloud"].shape == (3, 120, 3) and space["w"].shape == (3, 3, 32)
    # Each layer's cloud has a spread of 1, and projecting the states gives it back.
    spread = space["cloud"].pow(2).sum(-1).mean(-1).sqrt()
    assert torch.allclose(spread, torch.ones(3), atol=1e-6)
    assert torch.allclose(es.project(space, states), space["cloud"], atol=1e-5)
    path = es.path_for("tiny/model", tmp_path)
    es.save(space, path, "tiny/model", {"points": 120})
    loaded = es.load(path)
    assert torch.allclose(es.project(loaded, states), space["cloud"], atol=1e-3)
    assert es.load(tmp_path / "missing.json") is None


def test_the_alignment_keeps_each_layer_close_to_the_one_before():
    """A rotated copy of a layer, aligned by Procrustes, lands back on the layer before."""
    torch.manual_seed(0)
    first = torch.randn(200, 16)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    second = first @ q  # the same states, rotated in the full space
    space = es.fit(torch.stack([first, second]))
    assert torch.allclose(space["cloud"][0], space["cloud"][1], atol=1e-4)


def test_the_build_publishes_each_models_cloud_once(tmp_path, monkeypatch):
    """The answer pages fetch one cloud per model, not one per page (#4929)."""
    import json
    from types import SimpleNamespace

    from openinterpretability.builder.explanation_pages import write_space

    source = tmp_path / "data" / "x--m.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"layers": 2, "cloud": [[[0, 0, 0]], [[1, 1, 1]]],
                                  "mean": [[0]], "w": [[[0]]]}))  # fmt: skip
    monkeypatch.setattr(es, "path_for", lambda model, root=None: source)
    render = SimpleNamespace(output_dir=tmp_path / "site")
    href = write_space(render, "x/m")
    assert href == "models/x--m/embedding-space.json"
    published = json.loads((tmp_path / "site" / href).read_text())
    assert published == {"layers": 2, "cloud": [[[0, 0, 0]], [[1, 1, 1]]]}  # no mean or w
    monkeypatch.setattr(es, "path_for", lambda model, root=None: tmp_path / "none.json")
    assert write_space(render, "y/m") is None
