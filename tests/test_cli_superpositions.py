"""``open-interpretability superpositions``: grammatical-number and a missing definitions file."""

from __future__ import annotations

import json

from click.testing import CliRunner

from openinterpretability.audit_options import read_superposition
from openinterpretability.cli import cli
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)


def test_grammatical_number_writes_one_definition_per_layer(
    qwen_runner,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    loads = []

    def load(*args):
        loads.append(args)
        return qwen_runner

    monkeypatch.setattr("openinterpretability.cli_superpositions._load_runner", load)
    out = tmp_path / "defs/number.jsonl"
    args = ["superpositions", "grammatical-number", "--model", "o/m", "--layers", "0, 2"]
    result = CliRunner().invoke(cli, [*args, "--k", "4", "--n", "16", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert loads == [("o/m", "auto", "float32", None)]
    assert "layer 0: held-out accuracy " in result.output
    assert "layer 2: held-out accuracy " in result.output
    assert f"wrote 2 definitions → {out}" in result.output
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["layer"] for line in lines] == [0, 2]
    single = tmp_path / "one.json"
    single.write_text(lines[1] + "\n", encoding="utf-8")
    unit, model, layer, profile = read_superposition(single, "neuron")
    assert (unit, model, layer) == ("superposition", "o/m", 2)
    assert len(profile.members) == 4


def test_grammatical_number_rejects_layers_that_are_not_integers(tmp_path, monkeypatch):
    def load(*args):
        raise AssertionError("no model loads when the layers are bad")

    monkeypatch.setattr("openinterpretability.cli_superpositions._load_runner", load)
    args = ["superpositions", "grammatical-number", "--model", "o/m", "--layers", "0,x"]
    result = CliRunner().invoke(cli, [*args, "--out", str(tmp_path / "o.jsonl")])
    assert result.exit_code == 2
    assert "give layers as integers, e.g. 0,4,19" in result.output
    assert "--layers" in result.output
    assert not (tmp_path / "o.jsonl").exists()


def test_show_without_a_definitions_file_fails_and_names_the_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ["superpositions", "show", "a" * 12, "--model", "o/m", "--out", "s.json"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert "no definitions file at data/superpositions/o--m.jsonl" in result.output
    assert not (tmp_path / "s.json").exists()
