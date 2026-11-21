"""`open-interpretability activations` and `scan-cache` options (cli_scan.py)."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterpretability.cli import cli
from tests.test_agents import TEXTS

BASE = ["activations", "--model", "toy/model", "--layer", "1", "--neuron", "5"]


@pytest.fixture
def toy_cli(tmp_path, monkeypatch, toy_runner):
    toy_runner.model_name = "toy/model"
    toy_runner.model.config = SimpleNamespace()
    monkeypatch.setattr("openinterpretability.cli._load_runner", lambda *_a, **_k: toy_runner)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(TEXTS))
    return toy_runner


def invoke(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_stride_without_window_tokens_is_a_usage_error(toy_cli):
    result = invoke(*BASE, "--corpus", "corpus.txt", "--stride", 2)
    assert result.exit_code == 2
    assert "--stride needs --window-tokens, and can't exceed it." in result.output
    result = invoke(*BASE, "--corpus", "corpus.txt", "--window-tokens", 2, "--stride", 3)
    assert result.exit_code == 2


def test_out_writes_the_whole_text_tokens_at_or_above_min_strength(toy_cli, tmp_path):
    every = invoke(*BASE, "--corpus", "corpus.txt", "--max-tokens", 0, "--top", 0,
                   "--format", "jsonl")  # fmt: skip
    strengths = sorted(json.loads(line)["strength"] for line in every.stdout.splitlines())
    assert len(strengths) == sum(len(t) for t in TEXTS)  # max-tokens 0: every text whole
    distinct = sorted(set(strengths))
    middle = len(distinct) // 2
    floor = (distinct[middle - 1] + distinct[middle]) / 2  # between two shown strengths
    out = tmp_path / "rows" / "acts.jsonl"
    result = invoke(*BASE, "--corpus", "corpus.txt", "--max-tokens", 0, "--top", 0,
                    "--min-strength", floor, "--format", "jsonl", "--no-scan-cache",
                    "--out", out)  # fmt: skip
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == sum(s > floor for s in strengths)
    assert all(row["strength"] > floor for row in rows)
    assert f"Wrote {len(rows):,} tokens to {out}" in result.stderr
    assert "scan:" not in result.stderr  # no cache, so nothing cached or replayed
    assert result.stdout == ""


def test_scan_cache_says_when_it_is_off(monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_SCAN_CACHE", "off")
    result = invoke("scan-cache")
    assert result.exit_code == 0
    assert result.output == "The scan cache is off ($OPENINTERPRETABILITY_SCAN_CACHE).\n"
