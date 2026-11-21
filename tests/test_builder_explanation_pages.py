"""Explanation and answer pages (openinterpretability/builder/explanation_pages.py)."""

import json
from types import SimpleNamespace

from openinterpretability import embedding_space as es
from openinterpretability.builder.explanation_pages import write_space


def test_a_cloud_already_published_is_not_written_again(tmp_path, monkeypatch):
    source = tmp_path / "data" / "x--m.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"layers": 2, "cloud": [[[1, 2, 3]]]}))
    monkeypatch.setattr(es, "path_for", lambda model, root=None: source)
    render = SimpleNamespace(output_dir=tmp_path / "site")
    target = render.output_dir / "models" / "x--m" / "embedding-space.json"
    target.parent.mkdir(parents=True)
    target.write_text("already published", encoding="utf-8")

    assert write_space(render, "x/m") == "models/x--m/embedding-space.json"
    assert target.read_text(encoding="utf-8") == "already published"
