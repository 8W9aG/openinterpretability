"""The pole-effects command (cli_pole_effects.py)."""

import json
from pathlib import Path

from click.testing import CliRunner

from openinterpretability import __version__
from openinterpretability.cli import cli
from openinterpretability.planner_effects import table_path
from tests.conftest import N_LAYERS


def _corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    texts = ["abcdefghijklmnopqrstuvwxyz" * 2 + str(i) for i in range(8)]
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    return corpus


def _invoke(monkeypatch, toy_runner, *args):
    loads = []
    monkeypatch.setattr(
        "openinterpretability.cli._load_runner",
        lambda name, device, dtype: loads.append((name, device, dtype)) or toy_runner,
    )
    result = CliRunner().invoke(
        cli,
        ["pole-effects", "--model", "toy/model", "--windows", "2", "--window-tokens", "16",
         "--batch", "2", "--top", "5", "--device", "cpu", *map(str, args)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert loads == [("toy/model", "cpu", "float32")]
    return result.output


def test_pole_effects_writes_the_table_and_checks_poles_exactly(tmp_path, monkeypatch, toy_runner):
    out = tmp_path / "effects" / "toy.json"
    output = _invoke(
        monkeypatch, toy_runner, "--corpus", _corpus(tmp_path), "--check", "4", "--out", out
    )
    table = json.loads(out.read_text())
    assert table["model"] == "toy/model" and table["windows"] == 2 and table["seed"] == 0
    assert table["openinterpretability"] == __version__
    assert table["method"] == "attribution patching of single-pole mean ablation"
    width = toy_runner.model.model.layers[0].mlp.down_proj.in_features
    assert table["poles_total"] == 2 * N_LAYERS * width
    assert len(table["poles"]) == 5
    check = table["check"]
    assert check["poles"] == 4 and len(check["rows"]) == 4
    assert -1.0 <= check["spearman"] <= 1.0 and check["clean_loss"] > 0
    for row in check["rows"]:
        assert row["pole"] in ("positive", "negative") and 0 <= row["layer"] < N_LAYERS
        assert set(row) == {"layer", "neuron", "pole", "estimate", "exact"}
    rho = check["spearman"]
    assert output.strip() == f"Wrote 5 poles to {out}; Spearman {rho:.2f} on 4 checked poles"


def test_pole_effects_without_a_check_writes_to_the_planners_path(
    tmp_path, monkeypatch, toy_runner
):
    corpus = _corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = _invoke(monkeypatch, toy_runner, "--corpus", corpus, "--check", "0")
    out = table_path(Path("."), "toy/model")
    table = json.loads(out.read_text())
    assert table["check"] == {}
    assert output.strip() == f"Wrote 5 poles to {out}"
