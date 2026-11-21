"""The sides of an interference comparison: forced-open patches and reused noise."""

from __future__ import annotations

import pytest
import torch

from openinterpretability.interference.floors import Patched, Perturbed
from openinterpretability.patches.history import replay
from openinterpretability.patches.schema import Op, Patch
from openinterpretability.patches.tensors import put_tensor
from tests.test_interference import BASE, _runner


@pytest.fixture(autouse=True)
def tensor_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_EVIDENCE_DIR", str(tmp_path / "evidence"))


def test_forced_open_sets_force_on_gates_and_leaves_other_ops(tmp_path):
    gate = Op(op_id="g", type="gate", params={"function": {"kind": "token_match", "tokens": [5]}})
    steer = Op(
        op_id="s",
        type="steer",
        targets=["blocks.0.hook_resid_post"],
        gate="g",
        params={"vector": put_tensor(torch.randn(32))},
    )
    patch = Patch(base=BASE, ops=[gate, steer]).sealed()
    ops = replay(patch.id, {patch.id: patch}.__getitem__)
    side = Patched(_runner(), ops, "patch", tmp_path)
    opened = side.forced_open()
    (g_ref, g_op), (s_ref, s_op) = opened.ops
    assert g_op.params["force"] == "open" and g_op.params["function"]["tokens"] == [5]
    assert s_op == ops[1][1] and "force" not in s_op.params
    assert (opened.name, opened.root, opened.runner) == ("patch", tmp_path, side.runner)
    assert "force" not in side.ops[0][1].params  # the original is unchanged


def test_a_perturbed_side_draws_its_noise_once_and_reuses_it():
    base, other = _runner(0), _runner(1)
    side = Perturbed(base, other, "random")
    with side.entered() as runner:
        first = {k: v.clone() for k, v in runner.model.state_dict().items()}
    noise = side._noise
    with side.entered() as runner:
        second = {k: v.clone() for k, v in runner.model.state_dict().items()}
    assert side._noise is noise
    assert all(torch.equal(first[k], second[k]) for k in first)
