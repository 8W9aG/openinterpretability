"""The map command, with the sentence embedder replaced (cli_map.py)."""

import json
import sys
import types

import numpy as np
from click.testing import CliRunner

from openinterpretability import concept_map as cm
from openinterpretability.builder.wiki import write_experiment_page
from openinterpretability.cli import cli
from openinterpretability.planner import configured_models
from tests.test_circuits import at
from tests.test_dead import PROVENANCE


def _embedder(monkeypatch, made):
    """Put a fake ``sentence_transformers`` in place: two axes from each text's length."""

    class SentenceTransformer:
        def __init__(self, name, device):
            made.append((name, device))

        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings
            return np.array([[float(len(t)), 1.0] for t in texts])

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_without_the_map_extra_the_command_says_how_to_install_it(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import fails
    result = CliRunner().invoke(cli, ["map", "--wiki-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "pip install open-interpretability[map]" in result.output


def test_a_map_needs_more_poles_than_the_neighbours(monkeypatch, tmp_path):
    _embedder(monkeypatch, made := [])
    result = CliRunner().invoke(cli, ["map", "--wiki-dir", str(tmp_path / "wiki")])
    assert result.exit_code == 1
    assert "0 studied poles; a map needs more" in result.output and made == []


def test_a_later_run_keeps_every_old_point(monkeypatch, tmp_path, record):
    _embedder(monkeypatch, made := [])
    sub = record.submission.model_copy(update={"provenance": PROVENANCE})
    record = record.model_copy(update={"submission": sub})
    wiki = tmp_path / "wiki"
    count = cm.NEIGHBOURS + 2
    for n in range(count):
        write_experiment_page(at(record, 3, n), wiki)
    poles = cm.studied_poles(set(configured_models()), wiki)
    assert len(poles) == count
    old = {"version": 3, "fitted": "2026-11-01", "centre": [0.0, 0.0],
           "points": [{"slug": p.slug, "x": i / 20, "y": 0.5, "cluster": i % 2}
                      for i, p in enumerate(poles)]}  # fmt: skip
    out = tmp_path / "shared.json"
    out.write_text(json.dumps(old))
    result = CliRunner().invoke(cli, ["map", "--wiki-dir", str(wiki), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert made == [(cm.EMBEDDER, "cpu")]
    found = cm.load(out)
    assert found["version"] == 3 and found["fitted"] == "2026-11-01"
    assert [(p["slug"], p["x"]) for p in found["points"]] == [
        (p["slug"], p["x"]) for p in old["points"]
    ]
    regions = len(found["clusters"])
    named = sum(c["verified"] for c in found["clusters"])
    assert result.output == (
        f"{count} poles (0 new), {regions} regions ({named} named), version 3 → {out}\n"
    )
