"""The interference report on a tiny GPT-2 (openinterpretability/interference, #5230)."""

from __future__ import annotations

import pytest
import torch

from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.interference import report
from openinterpretability.interference.floors import Patched, Perturbed, Plain, random_patch
from openinterpretability.interference.run import Inputs, run
from openinterpretability.patches.history import replay
from openinterpretability.patches.schema import Op, Patch
from openinterpretability.patches.tensors import put_tensor
from tests.conftest import VOCAB, FakeTokenizer

BASE = "openai-community/gpt2@abc"


@pytest.fixture(autouse=True)
def tensor_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_EVIDENCE_DIR", str(tmp_path / "evidence"))


def _runner(seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=64, vocab_size=VOCAB)
    return NeuronHookRunner(GPT2LMHeadModel(config).eval(), FakeTokenizer())


def _report(runner, ops, tmp_path):
    noise, root = random_patch(ops, tmp_path / "noise")
    sides = {"base": Plain(runner, "base"), "floor:self": Plain(runner, "self"),
             "changed": Patched(runner, ops, "patch"),
             "floor:random": Patched(runner, noise, "random", root)}  # fmt: skip
    torch.manual_seed(1)
    batches = [torch.randint(1, VOCAB, (2, 16))]
    return run(sides, Inputs([], batches, [], 2, set(), report.edited(ops)))


def _ops(scale: float, column: int = 3):
    op = Op(op_id="w", type="sparse_add", targets=["blocks.1.mlp.W_out"],
            params={"axis": "col", "index": [column], "values": put_tensor(torch.randn(32, 1)),
                    "scale": scale})  # fmt: skip
    patch = Patch(base=BASE, ops=[op]).sealed()
    return replay(patch.id, {patch.id: patch}.__getitem__)


def test_a_zero_scale_patch_moves_nothing(tmp_path):
    rows = _report(_runner(), _ops(0.0), tmp_path)
    summary = rows[0]
    assert summary["verdict"] == "no known interference" and summary["footprint"] == 0
    assert summary["token_kl"]["changed"] == 0.0  # the patch at zero strength is the base


def test_an_edited_output_column_is_in_the_footprint(tmp_path):
    rows = _report(_runner(), _ops(1.0, column=3), tmp_path)
    feet = [r for r in rows if r["table"] == "footprint"]
    edited = [r for r in feet if r["layer"] == 1 and r["neuron"] == 3]
    assert edited and edited[0]["edited"] and edited[0]["output_change"] > 0
    assert rows[0]["verdict"] == "footprint not yet studied"  # no unit is studied
    assert rows[0]["token_kl"]["changed"] > 0 and "floor:random" in rows[0]["token_kl"]


def test_the_verdict_names_interference_before_unstudied_units():
    assert report.verdict(["a/L1/N2-positive"], [{"layer": 1}]) == "interference"
    assert report.verdict([], [{"layer": 1}]) == "footprint not yet studied"
    assert report.verdict([], []) == "no known interference"


def test_a_counted_flip_needs_no_floor_to_flip():
    def passes(ok):
        return {"passes": ok}

    flips = report._flips({"base": passes(True), "changed": passes(False),
                           "floor:self": passes(True), "floor:random": passes(True)})  # fmt: skip
    assert flips["counts"]
    noisy = report._flips({"base": passes(True), "changed": passes(False),
                           "floor:self": passes(True), "floor:random": passes(False)})  # fmt: skip
    assert noisy["flips"]["changed"] and not noisy["counts"]


def test_a_perturbed_pair_restores_the_base_weights():
    base, other = _runner(0), _runner(1)
    before = {k: v.clone() for k, v in base.model.state_dict().items()}
    with Perturbed(base, other, "random").entered() as runner:
        moved = any(not torch.equal(v, before[k]) for k, v in runner.model.state_dict().items())
    assert moved
    assert all(torch.equal(v, before[k]) for k, v in base.model.state_dict().items())


def test_edited_reads_the_neurons_a_patch_targets():
    ops = _ops(1.0, column=5)
    assert report.edited(ops) == {(1, 5)}
    dense = [("p:d", Op(op_id="d", type="dense_add", targets=["blocks.0.mlp.W_in"],
                        params={"delta": [[0.0]]}))]  # fmt: skip
    assert report.edited(dense) == {(0, None)}


def test_the_command_takes_exactly_one_changed_model():
    from click.testing import CliRunner

    from openinterpretability.cli import cli

    for extra in ([], ["--changed", "x", "--patch", "y"]):
        out = CliRunner().invoke(cli, ["interference", "--model", "m", "--out", "o.jsonl", *extra])
        assert out.exit_code != 0 and "exactly one" in out.output


def test_a_gated_patch_reports_kl_where_the_gate_opens_and_shuts(tmp_path):
    """Local scope: the KL on tokens where the gate opens, and where it stays shut."""
    gate = Op(op_id="g", type="gate", params={"function": {"kind": "token_match", "tokens": [5]}})
    steer = Op(op_id="s", type="steer", targets=["blocks.0.hook_resid_post"], gate="g",
               params={"vector": put_tensor(torch.randn(32) * 5)})  # fmt: skip
    patch = Patch(base=BASE, ops=[gate, steer]).sealed()
    ops = replay(patch.id, {patch.id: patch}.__getitem__)
    rows = _report(_runner(), ops, tmp_path)
    (local,) = rows[0]["gated_kl"].values()
    assert 0 < local["open_share"] < 1
    assert local["open_kl"] > 0 and local["shut_kl"] >= 0


def test_a_large_footprint_is_written_per_layer():
    from openinterpretability.interference.run import FOOTPRINT_ROWS, compact

    feet = [{"table": "footprint", "layer": i % 3, "neuron": i, "studied": i == 7, "edited": False}
            for i in range(FOOTPRINT_ROWS + 5)]  # fmt: skip
    out = compact([{"table": "summary"}, *feet])
    layers = [r for r in out if r["table"] == "footprint_layer"]
    assert sum(r["moved"] for r in layers) == FOOTPRINT_ROWS + 5 and len(layers) == 3
    assert [r["neuron"] for r in out if r["table"] == "footprint"] == [7]  # studied ones stay
    small = [{"table": "summary"}, *feet[:3]]
    assert compact(small) == small
