"""The wiring-backfill command (cli_wiring.py)."""

from click.testing import CliRunner

from openinterpretability.builder.wiki import write_experiment_page
from openinterpretability.cli import cli
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.wiring_backfill import load_rows
from tests.conftest import FakeTokenizer
from tests.test_circuits import at
from tests.test_dead import PROVENANCE


def test_backfill_loads_each_records_pinned_model_and_sorts_the_rows(
    tmp_path, monkeypatch, record, tiny_gpt2
):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    loads = []

    def load(name, device, dtype, revision=None, attn=None):
        loads.append((name, device, dtype, revision, attn))
        return runner

    monkeypatch.setattr("openinterpretability.cli_wiring._load_runner", load)
    pinned = PROVENANCE.model_copy(update={"dtype": "bfloat16", "model_revision": "abc123"})
    with_prov = at(record, 1, 7)
    with_prov = with_prov.model_copy(
        update={"submission": with_prov.submission.model_copy(update={"provenance": pinned})}
    )
    wiki = tmp_path / "wiki"
    for rec in (at(record, 1, 5), with_prov):
        write_experiment_page(rec, wiki)
    out = tmp_path / "wiring.jsonl"
    args = ["wiring-backfill", "--wiki-dir", str(wiki), "--out", str(out), "--device", "cpu"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert f"Wired 2 experiments → {out}" in result.output
    model = record.submission.identifier.model_name
    assert sorted(loads, key=str) == sorted(
        [(model, "cpu", "float32", None, None), (model, "cpu", "bfloat16", "abc123", None)],
        key=str,
    )
    rows = load_rows(out)
    assert len(rows) == 2 and list(rows) == sorted(rows)


def test_a_backfill_with_nothing_to_wire_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("openinterpretability.cli_wiring._load_runner", lambda *a, **k: None)
    out = tmp_path / "wiring.jsonl"
    args = ["wiring-backfill", "--wiki-dir", str(tmp_path / "wiki"), "--out", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert f"Wired 0 experiments → {out}" in result.output and not out.exists()
