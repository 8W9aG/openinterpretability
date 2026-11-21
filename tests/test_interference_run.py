"""The interference report's unit and route rows, on a tiny GPT-2 (#5230)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from openinterpretability import explain as ex
from openinterpretability.core.schemas import AdversarialTestResult, NeuronIdentifier
from openinterpretability.interference import report
from openinterpretability.interference.floors import Patched, Plain, random_patch
from openinterpretability.interference.run import Inputs, run
from openinterpretability.patches.history import replay
from openinterpretability.patches.schema import Op, Patch
from openinterpretability.patches.tensors import put_tensor
from tests.conftest import VOCAB
from tests.test_interference import BASE, _runner

TEXTS = ["abcdefgh ijklmnop", "the cat sat on the mat", "hello there, general"]


@pytest.fixture(autouse=True)
def tensor_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)


def _gated_column_edit():
    """A weight edit on column 3 of layer 1's output, gated on the residual stream."""
    reads = {"kind": "threshold", "input": "blocks.0.hook_resid_post", "index": 0}
    gate = Op(op_id="g", type="gate", params={"function": reads})
    params = {"axis": "col", "index": [3], "values": put_tensor(torch.randn(32, 1))}
    edit = Op(op_id="w", type="sparse_add", targets=["blocks.1.mlp.W_out"], gate="g", params=params)
    patch = Patch(base=BASE, ops=[gate, edit]).sealed()
    return replay(patch.id, {patch.id: patch}.__getitem__)


def _verified(record, neuron=3):
    ident = NeuronIdentifier(model_name="tiny", layer=1, neuron_idx=neuron, pole="positive")
    probes = [AdversarialTestResult(kind=kind, prompt=text, passed=True, expected_active=active)
              for kind, text, active in [("positive_holdout", "abc abc abc", True),
                                         ("hard_negative", "xyz xyz", False)]]  # fmt: skip
    sub = record.submission.model_copy(
        update={"identifier": ident, "max_activation": 1.0, "first_token_floor": False}
    )
    verdict = record.verdict.model_copy(update={"adversarial_test_results": probes})
    return record.model_copy(update={"submission": sub, "verdict": verdict})


def test_a_verified_pole_and_an_explained_prompt_get_rows_on_every_side(record, tmp_path):
    runner = _runner()
    ops = _gated_column_edit()
    noise, root = random_patch(ops, tmp_path / "noise")
    sides = {"base": Plain(runner, "base"), "floor:self": Plain(runner, "self"),
             "changed": Patched(runner, ops, "patch"),
             "floor:random": Patched(runner, noise, "random", root)}  # fmt: skip
    torch.manual_seed(1)
    batches = [torch.randint(1, VOCAB, (2, 16))]
    inputs = SimpleNamespace(explained=["L1/N3+"], token_id=5)
    explanation = SimpleNamespace(prompt="abcd", position=3, inputs=inputs)
    given = Inputs([_verified(record)], batches, [], 2, set(), report.edited(ops),
                   explanations=[explanation], texts=TEXTS, means_tokens=40,
                   key="tiny")  # fmt: skip
    rows = run(sides, given)
    summary = rows[0]
    (unit,) = [r for r in rows if r["table"] == "unit"]
    assert unit["slug"] == "tiny/L1/N3-positive" and (unit["layer"], unit["neuron"]) == (1, 3)
    assert set(unit["readings"]) == set(sides)
    assert -1.0 <= unit["logit_cosine"] <= 1.0
    assert unit["output_change"] > 0  # the edit read with its gate forced open moves column 3
    assert summary["poles_tested"] == 1
    (route,) = [r for r in rows if r["table"] == "route"]
    assert route["kind"] == "explanation" and set(route["coverage"]) == set(sides)
    assert route["drop"]["floor:self"] == pytest.approx(0.0, abs=1e-9)
    assert summary["routes"] == 1
