"""``open-interpretability interference`` from the command line, on a tiny GPT-2 (#5230)."""

from __future__ import annotations

import json

import pytest
import torch
from click.testing import CliRunner

from openinterpretability import cli_interference
from openinterpretability import explain as ex
from openinterpretability.cli import cli
from openinterpretability.interference import keep
from openinterpretability.patches.schema import Op, Patch
from openinterpretability.patches.tensors import put_tensor
from tests.test_interference import BASE, _runner

MODEL = BASE.split("@")[0]


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr(keep, "WINDOW", 16)  # the tiny GPT-2 holds 64 positions


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    texts = ["the cat sat on the mat and looked out", "hello there, general kenobi"] * 4
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    return path


def _loads(monkeypatch, **runners):
    """Serve each model name its own tiny runner, in place of a download."""
    loaded = []

    def load(name, device, dtype, revision=None):
        loaded.append(name)
        return runners[name]

    monkeypatch.setattr(cli_interference, "_load_runner", load)
    return loaded


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_another_model_is_read_against_the_base_and_a_random_change(tmp_path, monkeypatch, corpus):
    loaded = _loads(monkeypatch, **{MODEL: _runner(0), "other/model": _runner(1)})
    out = tmp_path / "report" / "rows.jsonl"
    args = [
        "interference",
        "--model",
        MODEL,
        "--changed",
        "other/model",
        "--out",
        str(out),
        "--corpus",
        str(corpus),
        "--tokens",
        "32",
        "--wiki-dir",
        str(tmp_path / "wiki"),
    ]
    result = CliRunner().invoke(cli, args, catch_exceptions=False)
    assert loaded == [MODEL, "other/model"]
    rows = _rows(out)
    summary = rows[0]
    assert summary["table"] == "summary" and json.loads(result.stdout) == summary
    assert summary["token_kl"]["changed"] > 0  # another model's weights move the tokens
    assert summary["token_kl"]["floor:random"] > 0
    feet = [r for r in rows if r["table"] == "footprint"]
    assert feet and not any(r["edited"] for r in feet)  # another model edits no neuron directly


def test_a_patch_is_read_against_a_random_patch_and_its_summary_is_the_evidence(
    tmp_path, monkeypatch, corpus
):
    loaded = _loads(monkeypatch, **{MODEL: _runner(0)})
    torch.manual_seed(2)
    params = {"axis": "col", "index": [3], "values": put_tensor(torch.randn(32, 1))}
    op = Op(op_id="w", type="sparse_add", targets=["blocks.1.mlp.W_out"], params=params)
    draft = tmp_path / "draft.json"
    draft.write_text(Patch(base=BASE, ops=[op]).model_dump_json())
    out, evidence = tmp_path / "rows.jsonl", tmp_path / "keep.json"
    args = ["interference", "--model", MODEL, "--patch", str(draft), "--evidence-out",
            str(evidence), "--out", str(out), "--corpus", str(corpus), "--tokens", "32",
            "--wiki-dir", str(tmp_path / "wiki")]  # fmt: skip
    result = CliRunner().invoke(cli, args, catch_exceptions=False)
    assert loaded == [MODEL]  # the changed side is the base with the patch applied
    rows = _rows(out)
    assert json.loads(evidence.read_text()) == rows[0] == json.loads(result.stdout)
    edited = [r for r in rows if r["table"] == "footprint" and r["edited"]]
    assert [(r["layer"], r["neuron"]) for r in edited] == [(1, 3)]
