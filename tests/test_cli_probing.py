"""`open-interpretability probe-compare`: its errors and a held-out source (cli_probing.py)."""

import json

import pytest
from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.core.probing import ProbeComparison
from openinterpretability.core.schemas import AuditRecord, NeuronIdentifier
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import N_LAYERS
from tests.test_probing import OffsetTokenizer

TEXTS = ["abcabcxyz" * 4, "zyxcba" * 5, "aaabbbccc" * 3, "xxyyzz" * 6] * 3


def _labels(path, letter="a"):
    rows = [{"text": t, "spans": [[i, i + 1] for i, c in enumerate(t) if c == letter]}
            for t in TEXTS]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch, toy_model, submission):
    runner = NeuronHookRunner(toy_model, OffsetTokenizer())
    monkeypatch.setattr("openinterpretability.cli._load_runner", lambda *_a, **_k: runner)
    ident = NeuronIdentifier(model_name="toy", layer=N_LAYERS - 1, neuron_idx=3, pole="positive")
    sub = submission.model_copy(update={"identifier": ident})
    record = tmp_path / "record.json"
    record.write_text(AuditRecord(submission=sub).model_dump_json())
    return tmp_path, record, sub


def _run(record, train, out, *extra):
    args = ["probe-compare", "--record", str(record), "--train", str(train)]
    args += ["--concept", "the letter a", "--window-tokens", "8", "--out", str(out), *extra]
    return CliRunner().invoke(cli, args)


def test_a_held_out_source_is_the_test_set(setup):
    tmp_path, record, _ = setup
    train, test = _labels(tmp_path / "train.jsonl"), _labels(tmp_path / "test.jsonl")
    out = tmp_path / "report.json"
    result = _run(record, train, out, "--test", str(test))
    assert result.exit_code == 0, result.output
    report = ProbeComparison.model_validate_json(out.read_text())
    assert report.split == "source" and report.labels == f"{train} → {test}"
    assert (report.train_source, report.test_source) == ("train.jsonl", "test.jsonl")


def test_a_record_with_no_scale_is_refused(setup):
    tmp_path, record, sub = setup
    record.write_text(
        AuditRecord(submission=sub.model_copy(update={"max_activation": None})).model_dump_json()
    )
    result = _run(record, _labels(tmp_path / "train.jsonl"), tmp_path / "report.json")
    assert result.exit_code != 0
    assert "probe-compare needs a neuron pole with a recorded scale." in result.output


def test_labels_of_one_class_are_an_error(setup):
    tmp_path, record, _ = setup
    train = _labels(tmp_path / "train.jsonl", letter="!")  # no span anywhere
    out = tmp_path / "report.json"
    result = _run(record, train, out)
    assert result.exit_code != 0 and not out.exists()
    assert "tokens hold only one class" in result.output
