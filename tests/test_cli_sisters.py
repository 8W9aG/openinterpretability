"""The ``open-interpretability sisters`` command: which model pairs it matches, and its output."""

from __future__ import annotations

import copy
import random

import pytest
from click.testing import CliRunner

from openinterpretability import cli_sisters
from openinterpretability.builder.wiki import archive_record
from openinterpretability.cli import cli
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.sisters import load_rows
from tests.conftest import FakeTokenizer
from tests.test_sisters import _record
from tests.test_wiring import qwen_runner

__all__ = ["qwen_runner"]  # the fixture, used by name below


@pytest.fixture
def registry(tmp_path, record):
    """A registry of two configured models, each with one VERIFIED pole, and a corpus."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["tiny", "tiny-twin"]\n'
    )
    wiki = tmp_path / "wiki"
    for rec in (_record(record, "tiny", 1, 5, "positive"),
                _record(record, "tiny-twin", 1, 5, "positive")):  # fmt: skip
        archive_record(rec, wiki)
    rng = random.Random(0)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(
        "".join(rng.choice("abcdefghijklmno") for _ in range(rng.randint(20, 40)))
        for _ in range(30)
    ))  # fmt: skip
    return wiki, corpus


@pytest.fixture
def loads(qwen_runner, monkeypatch):
    """Model loads give the tiny Qwen2 for "tiny" and an exact copy of it for any other name."""
    twin = NeuronHookRunner(copy.deepcopy(qwen_runner.model), FakeTokenizer())
    seen = []

    def load(name, device, dtype, revision=None):
        seen.append((name, device, dtype))
        return qwen_runner if name == "tiny" else twin

    monkeypatch.setattr(cli_sisters, "_load_runner", load)
    return seen


def _args(registry, out, *extra):
    wiki, corpus = registry
    return ["sisters", "--wiki-dir", str(wiki), "--corpus", str(corpus), "--out", str(out),
            "-k", "2", "--device", "cpu", *extra]  # fmt: skip


def test_one_models_poles_are_matched_and_a_rerun_resumes(registry, loads, tmp_path):
    out = tmp_path / "sisters.jsonl"
    result = CliRunner().invoke(cli, _args(registry, out, "--model", "tiny"))
    assert result.exit_code == 0, result.output
    assert f"Matched 1 poles of tiny against tiny-twin → {out}\n" in result.output
    (row,) = load_rows(out)["tiny/L1/N5-positive"]
    assert row.other == "tiny-twin" and row.mutual
    assert {name for name, _, _ in loads} == {"tiny", "tiny-twin"}
    assert all(device == "cpu" and dtype == "float32" for _, device, dtype in loads)

    again = CliRunner().invoke(cli, _args(registry, out, "--model", "tiny"))
    assert f"Matched 0 poles of tiny against tiny-twin → {out}\n" in again.output


def test_every_configured_pair_is_matched_by_default(registry, loads, tmp_path):
    out = tmp_path / "sisters.jsonl"
    result = CliRunner().invoke(cli, _args(registry, out))
    assert result.exit_code == 0, result.output
    assert "Matched 1 poles of tiny against tiny-twin" in result.output
    assert "Matched 1 poles of tiny-twin against tiny" in result.output
    lines = out.read_text().splitlines()
    assert lines == sorted(lines)  # sort_rows keeps a stable order
    assert set(load_rows(out)) == {"tiny/L1/N5-positive", "tiny-twin/L1/N5-positive"}


def test_other_limits_the_search_to_the_models_it_names(registry, loads, tmp_path):
    out = tmp_path / "sisters.jsonl"
    result = CliRunner().invoke(cli, _args(registry, out, "--model", "tiny", "--other", "copy"))
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == f"Matched 1 poles of tiny against copy → {out}"
    (row,) = load_rows(out)["tiny/L1/N5-positive"]
    assert row.other == "copy"


def test_a_model_that_is_not_configured_is_matched_with_its_siblings(registry, loads, tmp_path):
    out = tmp_path / "sisters.jsonl"
    result = CliRunner().invoke(cli, _args(registry, out, "--model", "Qwen/Qwen2.5-0.5B"))
    assert result.exit_code == 0, result.output
    # It has no VERIFIED pole here, so its sibling pair gets no rows.
    assert result.output.splitlines()[-1] == (
        f"Matched 0 poles of Qwen/Qwen2.5-0.5B against Qwen/Qwen2.5-1.5B → {out}"
    )
    assert load_rows(out) == {}


def test_no_model_pair_is_a_usage_error(registry, loads, tmp_path):
    out = tmp_path / "sisters.jsonl"
    result = CliRunner().invoke(cli, _args(registry, out, "--model", "alone/model"))
    assert result.exit_code == 2
    assert "no model pairs to match; pass --model and --other" in result.output
    assert loads == [] and not out.exists()


def test_k_below_one_is_refused(registry, loads, tmp_path):
    # #5890: k = 0 gave units no match and crashed the backward pass.
    out = tmp_path / "sisters.jsonl"
    args = _args(registry, out, "--model", "tiny")
    args[args.index("-k") + 1] = "0"
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2 and "Invalid value for '-k'" in result.output
    assert loads == [] and not out.exists()


def test_other_without_a_first_model_is_refused(registry, loads, tmp_path):
    # #5890: with no configured models and no --model, the pair was (None, other).
    (registry[0].parent / "pyproject.toml").write_text("[tool.openinterpretability]\nmodels = []\n")
    out = tmp_path / "sisters.jsonl"
    result = CliRunner().invoke(cli, _args(registry, out, "--other", "tiny"))
    assert result.exit_code == 2
    assert "no model pairs to match; pass --model and --other" in result.output
    assert loads == [] and not out.exists()
