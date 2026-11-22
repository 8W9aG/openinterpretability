"""Applying patch operations as hooks: each op gives what editing the model would (#5229)."""

from __future__ import annotations

import copy

import pytest
import torch

from openinterpretability.patches import history
from openinterpretability.patches.apply import ZeroStartError, logits, zero_start_check
from openinterpretability.patches.schema import Op, PatchError, ref
from openinterpretability.patches.tensors import put_tensor
from tests.test_patches import IDS, MLP, D, T, lookup_of, neox, patch, run, tensor_store

__all__ = ["neox", "tensor_store"]  # the fixtures, used by name below


def _run_ops(model, *ops, strength=1.0):
    p = patch(list(ops))
    return run(model, p.id, lookup_of(p), strength)


def _edited(model, edit):
    """The logits of a copy of ``model`` with ``edit`` applied to its weights."""
    other = copy.deepcopy(model)
    with torch.no_grad():
        edit(other)
    return logits(other, IDS)


def test_a_replace_op_gives_the_replaced_weights_logits(neox):
    value = torch.randn(D, MLP)
    op = Op(op_id="r", type="replace", targets=["blocks.1.mlp.W_out"],
            params={"value": put_tensor(value)})  # fmt: skip
    expected = _edited(neox, lambda m: m.gpt_neox.layers[1].mlp.dense_4h_to_h.weight.copy_(value))
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)


def test_a_scale_op_gives_the_scaled_weights_logits(neox):
    op = Op(op_id="s", type="scale", targets=["blocks.0.mlp.W_in"], params={"factor": 0.25})
    expected = _edited(neox, lambda m: m.gpt_neox.layers[0].mlp.dense_h_to_4h.weight.mul_(0.25))
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)


def _neurons(act=None):
    params = {"W_in": T(4, D), "b_in": T(4), "W_out": T(D, 4)} | ({"act": act} if act else {})
    return Op(op_id="n", type="add_neurons", targets=["blocks.0.mlp.hook_out"], params=params)


def test_added_neurons_use_the_named_activation_else_the_mlps_own(neox):
    torch.manual_seed(3)
    own, relu = _neurons(), _neurons("relu")
    relu.params.update({k: own.params[k] for k in ("W_in", "b_in", "W_out")})
    gelu = _neurons("gelu")
    gelu.params.update({k: own.params[k] for k in ("W_in", "b_in", "W_out")})
    base = logits(neox, IDS)
    by_own, by_gelu, by_relu = (_run_ops(neox, op) for op in (own, gelu, relu))
    assert torch.allclose(by_own, by_gelu, atol=1e-5)  # Pythia's MLP activation is GELU
    assert not torch.allclose(by_relu, by_gelu, atol=1e-4)
    assert not torch.allclose(by_relu, base, atol=1e-4)


def test_a_linear_module_with_zero_weights_adds_its_bias_like_a_steer(neox):
    vector = put_tensor(torch.randn(D))
    target = ["blocks.1.hook_resid_post"]
    module = {"kind": "linear", "W": T(D, D, zero=True), "b": vector}
    added = Op(op_id="m", type="add_module", targets=target, params={"module": module})
    steer = Op(op_id="v", type="steer", targets=target, params={"vector": vector})
    assert torch.allclose(_run_ops(neox, added), _run_ops(neox, steer), atol=1e-5)
    assert not torch.allclose(_run_ops(neox, added), logits(neox, IDS), atol=1e-4)


def test_a_module_can_read_another_hook_point_than_it_writes(neox):
    module = {"kind": "linear", "W": T(D, D)}
    target = ["blocks.1.hook_resid_post"]
    own = Op(op_id="m", type="add_module", targets=target, params={"module": module})
    other = Op(op_id="m", type="add_module", targets=target,
               params={"module": module, "input": "blocks.0.hook_resid_post"})  # fmt: skip
    from_target, from_input = _run_ops(neox, own), _run_ops(neox, other)
    assert not torch.allclose(from_target, from_input, atol=1e-4)
    assert not torch.allclose(from_input, logits(neox, IDS), atol=1e-4)


def test_a_module_retuned_to_an_unknown_kind_fails_when_the_model_runs(neox):
    # The schema refuses the kind in a new op, but a set op's merge is not checked again.
    op = Op(op_id="m", type="add_module", targets=["blocks.1.hook_resid_post"],
            params={"module": {"kind": "linear", "W": T(D, D)}})  # fmt: skip
    first = patch([op])
    setter = Op(op_id="k", type="set", targets=[ref(first.id, "m")],
                params={"params": {"module": {"kind": "conv"}}})  # fmt: skip
    second = patch([setter], parents=[first.id])
    with pytest.raises(PatchError, match="unknown module kind 'conv'"):
        run(neox, second.id, lookup_of(first, second))


def _clamp(index, value):
    return Op(op_id="c", type="clamp", targets=["blocks.0.mlp.hook_post"],
              params={"index": index, "value": value})  # fmt: skip


@pytest.mark.parametrize(("params", "value"), [
    ({"unit": "neuron", "index": 5}, 0.0),
    ({"unit": "neuron", "index": 5, "mode": "mean", "value": 2.0}, 2.0),
    ({"unit": "neuron", "index": 5, "mode": "zero", "value": 2.0}, 0.0),
])  # fmt: skip
def test_pruning_a_neuron_sets_it_to_zero_or_its_mean(neox, params, value):
    op = Op(op_id="p", type="prune", targets=["blocks.0.mlp.hook_out"], params=params)
    assert torch.allclose(_run_ops(neox, op), _run_ops(neox, _clamp(5, value)), atol=1e-6)
    assert not torch.allclose(_run_ops(neox, op), logits(neox, IDS), atol=1e-6)


def test_pruning_a_head_is_zeroing_its_slice_of_the_output_weights(neox):
    op = Op(op_id="p", type="prune", targets=["blocks.0.attn.hook_z"],
            params={"unit": "head", "index": 1})  # fmt: skip
    width = D // 2  # two heads

    def zero_head(model):
        model.gpt_neox.layers[0].attention.dense.weight[:, width:] = 0.0

    assert torch.allclose(_run_ops(neox, op), _edited(neox, zero_head), atol=1e-5)


def test_an_operation_that_is_not_built_yet_is_refused(neox):
    op = Op(op_id="h", type="add_head", targets=["blocks.0.attn.hook_z"], params={})
    with pytest.raises(PatchError, match="operation type 'add_head' is in the registry"):
        _run_ops(neox, op)


def test_the_zero_start_check_fails_a_patch_that_moves_the_logits(neox):
    op = Op(op_id="v", type="steer", targets=["blocks.1.hook_resid_post"], params={"vector": T(D)})
    p = patch([op])
    replayed = history.replay(p.id, lookup_of(p))
    assert zero_start_check(neox, replayed, IDS) == 0.0
    with pytest.raises(ZeroStartError, match="patch at strength 1.0 moves a logit by"):
        zero_start_check(neox, replayed, IDS, strength=1.0)


def test_a_dense_add_op_gives_the_added_weights_logits(neox):
    delta = torch.randn(D, MLP)
    op = Op(op_id="d", type="dense_add", targets=["blocks.1.mlp.W_out"],
            params={"delta": put_tensor(delta)})  # fmt: skip
    expected = _edited(neox, lambda m: m.gpt_neox.layers[1].mlp.dense_4h_to_h.weight.add_(delta))
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)
