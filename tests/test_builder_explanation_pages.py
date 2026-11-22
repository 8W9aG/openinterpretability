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


def test_the_view_counts_the_landed_and_the_pending_experiments_of_its_run():
    from datetime import UTC, datetime

    from openinterpretability.builder.explanation_pages import ExplanationView
    from openinterpretability.core.explanation_records import ExplainRun
    from tests.test_explanation_records import _record

    record = _record(datetime(2026, 11, 17, tzinfo=UTC))
    pending = ["b VERIFIED u2", "c REJECTED u3"]
    run = ExplainRun(share_before=0.1, share_after=0.4, stopped="x", landed=["a VERIFIED u1"],
                     pending=pending, local=["d VERIFIED wiki/r.json"])  # fmt: skip
    view = ExplanationView(record=record.model_copy(update={"run": run}), source="x.json")
    assert (view.landed, view.pending, view.local) == (1, 2, 1)
    bare = ExplanationView(record=record, source="x.json")
    assert (bare.landed, bare.pending, bare.local) == (0, 0, 0)
