"""Superpositions and circuits read for ``explain`` (explain_groups.py, #3617)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from openinterpretability import explain_groups as eg
from openinterpretability.planner_superpositions import definition_identifier


def test_a_model_without_a_superposition_file_has_none(tmp_path):
    assert eg.superpositions("org/model", tmp_path) == []


def test_superpositions_are_read_from_the_models_file(tmp_path):
    definition = {"model": "org/model", "layer": 3, "method": "sparse-probe", "source": "x",
                  "members": [{"neuron_idx": 1, "weight": 0.5},
                              {"neuron_idx": 4, "weight": -0.2}]}  # fmt: skip
    (tmp_path / "org--model.jsonl").write_text(json.dumps(definition) + "\n\n")
    (group,) = eg.superpositions("org/model", tmp_path)
    slug = definition_identifier(definition).slug
    assert (group.kind, group.slug, group.definition) == ("superposition", slug, definition)
    assert group.name == slug.split("/", 1)[1] and group.name.startswith("L3/S")
    assert group.neurons == ((3, 1, 0.5), (3, 4, -0.2)) and group.heads == ()


def test_a_circuit_folder_with_no_record_gets_no_label(tmp_path):
    source = next(Path("wiki/circuits").glob("*/C*/*.json"))
    model_dir = tmp_path / "circuits" / source.parent.parent.name
    (model_dir / source.parent.name).mkdir(parents=True)
    shutil.copy(source, model_dir / source.parent.name / source.name)
    (model_dir / "C999999").mkdir()  # a folder whose record is not written yet
    record = json.loads(source.read_text())
    labels = eg.circuit_labels(tmp_path, record["hypothesis"]["model_name"])
    assert list(labels) == [source.parent.name]
    assert eg.circuit_labels(tmp_path, "org/unknown") == {}  # no folder for the model
