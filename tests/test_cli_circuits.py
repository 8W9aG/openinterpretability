"""The ``open-interpretability circuits`` commands: identify, hubs, propose and review."""

import json

import pytest
from click.testing import CliRunner

from openinterpretability import circuit_ids, hubs
from openinterpretability import cli as cli_module
from openinterpretability.builder.wiki import archive_record
from openinterpretability.cli import cli
from openinterpretability.core import circuit_records as cr
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import FakeTokenizer
from tests.test_circuits import at, wiring

MODEL = "Qwen--Qwen2.5-0.5B"
PAIR = cr.PromptPair(clean="abcd", corrupted="xbcd", good="e", bad="f")
TASK = cr.CircuitTask(name="t", description="d", pairs=[PAIR])


@pytest.fixture
def tiny_runner(tiny_gpt2, monkeypatch):
    """Every model load returns the tiny GPT-2, so no model is downloaded."""
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    loads = []

    def load(model, device, dtype, revision=None):
        loads.append((model, device, dtype, revision))
        return runner

    monkeypatch.setattr(cli_module, "_load_runner", load)
    return loads


def _chain_wiki(record, wiki):
    """A wiki whose strong triangle L0/N1, L0/N2, L1/N3 is one cluster, and L1/N9 hangs off it."""
    down = [("mlp_neuron", 0, 2, 2.0), ("mlp_neuron", 1, 3, 2.0)]
    records = [
        at(record, 0, 1, activations=wiring("activations", down=down)),
        at(record, 0, 2, activations=wiring("activations", down=[("mlp_neuron", 1, 3, 2.0)])),
        at(record, 1, 3, activations=wiring("activations", down=[("mlp_neuron", 1, 9, 0.0)])),
        at(record, 1, 9),
    ]
    for rec in records:
        archive_record(rec, wiki)


def _hub_wiki(record, wiki):
    """A wiki where ten L0 units all wire to L1/N1, which makes L1/N1 a hub."""
    spoke = wiring("activations", down=[("mlp_neuron", 1, 1, 0.1)])
    for rec in [*(at(record, 0, i, activations=spoke) for i in range(10)), at(record, 1, 1)]:
        archive_record(rec, wiki)


def test_identify_names_a_new_circuit_then_reports_no_change(record, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no wiring rows file here
    wiki, out = tmp_path / "wiki", tmp_path / "circuits"
    _chain_wiki(record, wiki)
    args = ["circuits", "identify", "--wiki-dir", str(wiki), "--out-dir", str(out)]
    first = CliRunner().invoke(cli, args)
    assert first.exit_code == 0, first.output
    assert first.output == f"{MODEL}: 1 circuits (new)\n"
    circuits, ever = circuit_ids.load(circuit_ids.path_for(MODEL, out))
    assert ever == {"C1"} and len(circuits["C1"]) == 3

    second = CliRunner().invoke(cli, args)
    assert second.output == f"{MODEL}: 1 circuits (no change)\n"
    assert len(circuit_ids.path_for(MODEL, out).read_text().splitlines()) == 1


def test_identify_skips_models_other_than_the_one_asked_for(record, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wiki, out = tmp_path / "wiki", tmp_path / "circuits"
    _chain_wiki(record, wiki)
    args = ["circuits", "identify", "--wiki-dir", str(wiki), "--out-dir", str(out)]
    result = CliRunner().invoke(cli, [*args, "--model", "other--model"])
    assert result.exit_code == 0 and result.output == ""
    assert not out.exists()


def test_hubs_records_one_norm_row_per_hub(record, tmp_path, monkeypatch, tiny_runner):
    monkeypatch.chdir(tmp_path)
    wiki, out = tmp_path / "wiki", tmp_path / "hubs"
    _hub_wiki(record, wiki)
    args = ["circuits", "hubs", "--wiki-dir", str(wiki), "--out-dir", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    path = hubs.path_for(MODEL, out)
    assert result.output == f"{MODEL}: 1 hubs → {path}\n"
    (row,) = hubs.load_norms(path).values()
    assert row.hub == f"{MODEL}/L1/N1/HUB" and row.norm > 0
    assert tiny_runner == [("Qwen/Qwen2.5-0.5B", "cpu", "float32", None)]


def test_hubs_skips_models_without_hubs_and_models_not_asked_for(
    record, tmp_path, monkeypatch, tiny_runner
):
    monkeypatch.chdir(tmp_path)
    wiki, out = tmp_path / "wiki", tmp_path / "hubs"
    _chain_wiki(record, wiki)  # wired, but no unit reaches the hub bar
    args = ["circuits", "hubs", "--wiki-dir", str(wiki), "--out-dir", str(out)]
    assert CliRunner().invoke(cli, args).output == ""
    assert CliRunner().invoke(cli, [*args, "--model", "other--model"]).output == ""
    assert not out.exists() and tiny_runner == []


def _circuits_dir(tmp_path):
    folder = tmp_path / "circuits"
    members = [f"models/{MODEL}/L0/N1.html", f"models/{MODEL}/L1/N3.html"]
    circuit_ids.append(
        circuit_ids.path_for(MODEL, folder),
        [{"date": "2026-11-01", "event": "new", "id": "C1", "members": members}],
    )
    return folder


def _propose_args(tmp_path, circuit="C1"):
    task = tmp_path / "task.json"
    task.write_text(TASK.model_dump_json())
    return ["circuits", "propose", "--model", "Qwen/Qwen2.5-0.5B", "--circuit", circuit,
            "--task", str(task), "--explanation", "copies the letter", "--agent-model",
            "external/a", "--size", "3", "--circuits-dir", str(_circuits_dir(tmp_path)),
            "--device", "cpu", "--out", str(tmp_path / "hypothesis.json")]  # fmt: skip


def test_propose_writes_the_cluster_grown_to_its_size(tmp_path, tiny_runner):
    result = CliRunner().invoke(cli, _propose_args(tmp_path))
    assert result.exit_code == 0, result.output
    out = tmp_path / "hypothesis.json"
    assert result.output == f"3 members → {out}\n"
    hypothesis = cr.CircuitHypothesis.model_validate_json(out.read_text())
    assert hypothesis.circuit_id == "C1" and len(hypothesis.members) == 3
    assert {"L0/N1", "L1/N3"} <= set(hypothesis.members)
    assert hypothesis.explanation == "copies the letter" and hypothesis.task == TASK
    assert tiny_runner == [("Qwen/Qwen2.5-0.5B", "cpu", "float32", None)]


def test_propose_rejects_a_circuit_the_model_does_not_have(tmp_path, tiny_runner):
    result = CliRunner().invoke(cli, _propose_args(tmp_path, circuit="C9"))
    assert result.exit_code == 2
    assert "C9 is not a current circuit of Qwen/Qwen2.5-0.5B" in result.output
    assert not (tmp_path / "hypothesis.json").exists() and tiny_runner == []


def test_review_archives_the_decided_record(tmp_path, tiny_runner):
    hypothesis = cr.CircuitHypothesis(
        model_name="Qwen/Qwen2.5-0.5B", model_revision="abc", circuit_id="C1", task=TASK,
        members=["L0/N1", "L1/H0"], explanation="e", submitting_agent_model="external/a",
    )  # fmt: skip
    probes = cr.CircuitProbes(
        holdouts=[PAIR], negative_tasks=[TASK], reviewer_agent_model="external/b"
    )
    h_file, p_file = tmp_path / "h.json", tmp_path / "p.json"
    h_file.write_text(hypothesis.model_dump_json())
    p_file.write_text(probes.model_dump_json())
    wiki = tmp_path / "wiki"
    args = ["circuits", "review", "--hypothesis", str(h_file), "--probes", str(p_file),
            "--wiki-dir", str(wiki), "--device", "cpu"]  # fmt: skip
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    (path,) = (wiki / cr.CIRCUITS_WIKI / MODEL / "C1").glob("*.json")
    record = cr.CircuitRecord.model_validate_json(path.read_text())
    assert record.hypothesis == hypothesis
    gates = ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in record.verdict.gates.items())
    assert result.output == f"{record.verdict.status} ({gates}) → {path}\n"
    assert tiny_runner == [("Qwen/Qwen2.5-0.5B", "cpu", "float32", "abc")]
    assert json.loads(path.read_text())["hypothesis"]["circuit_id"] == "C1"
