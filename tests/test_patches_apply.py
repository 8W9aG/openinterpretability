"""Applying patch operations as hooks: each op gives what editing the model would (#5229)."""

from __future__ import annotations

import copy

import pytest
import torch

from openinterpretability.patches import history
from openinterpretability.patches.apply import ZeroStartError, applied, logits, zero_start_check
from openinterpretability.patches.schema import Op, PatchError, ref
from openinterpretability.patches.tensors import load, put_tensor
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


GATED = [1, 3]  # the positions of token 7 in IDS
OTHER = [0, 2, 4, 5]


def _raw(*ops):
    """``ops`` as one patch ``p`` would replay them, with their gates left as bare names."""
    return [(ref("p", op.op_id), op) for op in ops]


def _gate(force=None):
    params = {"function": {"kind": "token_match", "tokens": [7]}} | (
        {"force": force} if force else {}
    )
    return Op(op_id="g", type="gate", params=params)


def _logits(model, ops, strength=1.0, root=None, ids=IDS):
    with applied(model, ops, strength, root):
        return logits(model, ids)


def _layer_one_ops():
    """One op of each installed kind, all in the last block.

    So each position's logits depend only on the edit at that position.
    """
    torch.manual_seed(5)
    return {
        "weight": Op(op_id="w", type="dense_add", targets=["blocks.1.mlp.W_out"],
                     params={"delta": T(D, MLP)}),
        "bias": Op(op_id="w", type="dense_add", targets=["blocks.1.mlp.b_out"],
                   params={"delta": T(D)}),
        "neurons": Op(op_id="w", type="add_neurons", targets=["blocks.1.mlp.hook_out"],
                      params={"W_in": T(4, D), "W_out": T(D, 4)}),
        "layer": Op(op_id="w", type="add_layer", targets=["blocks.1.hook_resid_post"],
                    params={"W1": T(8, D), "W2": T(D, 8)}),
        "module": Op(op_id="w", type="add_module", targets=["blocks.1.hook_resid_post"],
                     params={"module": {"kind": "linear", "W": T(D, D)}}),
        "neuron prune": Op(op_id="w", type="prune", targets=["blocks.1.mlp.hook_out"],
                           params={"unit": "neuron", "index": 3}),
        "head prune": Op(op_id="w", type="prune", targets=["blocks.1.attn.hook_z"],
                         params={"unit": "head", "index": 0}),
        "clamp": Op(op_id="w", type="clamp", targets=["blocks.1.mlp.hook_post"],
                    params={"index": 3, "value": 5.0}),
        "steer": Op(op_id="w", type="steer", targets=["blocks.1.hook_resid_post"],
                    params={"vector": T(D)}),
    }  # fmt: skip


@pytest.mark.parametrize("kind", list(_layer_one_ops()))
def test_a_gate_named_by_its_bare_op_id_shuts_or_opens_each_kind_of_op(neox, kind):
    op = _layer_one_ops()[kind]
    base, full = logits(neox, IDS), _logits(neox, _raw(op))
    assert not torch.allclose(full, base, atol=1e-4)  # the op acts
    for name in ("g", ref("p", "g")):
        gated = op.model_copy(update={"gate": name})
        assert torch.allclose(_logits(neox, _raw(_gate("shut"), gated)), base, atol=1e-5)
        assert torch.allclose(_logits(neox, _raw(_gate("open"), gated)), full, atol=1e-5)


@pytest.mark.parametrize("kind", list(_layer_one_ops()))
def test_a_token_gate_applies_each_kind_of_op_only_where_it_is_open(neox, kind):
    op = _layer_one_ops()[kind]
    base, full = logits(neox, IDS), _logits(neox, _raw(op))
    gated = _logits(neox, _raw(_gate(), op.model_copy(update={"gate": "g"})))
    assert torch.allclose(gated[:, GATED], full[:, GATED], atol=1e-5)
    assert torch.allclose(gated[:, OTHER], base[:, OTHER], atol=1e-5)
    assert not torch.allclose(full[:, GATED], base[:, GATED], atol=1e-4)


def test_the_gates_are_read_again_on_each_forward_pass(neox):
    steer = _layer_one_ops()["steer"].model_copy(update={"gate": "g"})
    other = torch.tensor([[7, 3, 3, 3, 7, 7]])
    ops = _raw(_gate(), steer)
    with applied(neox, ops):
        first, second = logits(neox, IDS), logits(neox, other)
    assert torch.allclose(first, _logits(neox, ops), atol=1e-6)
    assert torch.allclose(second, _logits(neox, ops, ids=other), atol=1e-6)


def test_an_ops_scale_multiplies_the_strength(neox):
    vector = torch.randn(D)
    target = ["blocks.1.hook_resid_post"]
    scaled = Op(op_id="v", type="steer", targets=target,
                params={"vector": put_tensor(vector), "scale": 4.0})  # fmt: skip
    plain = Op(op_id="v", type="steer", targets=target, params={"vector": put_tensor(2 * vector)})
    assert torch.allclose(_logits(neox, _raw(scaled), strength=0.5), _logits(neox, _raw(plain)),
                          atol=1e-5)  # fmt: skip


def test_tensors_load_from_the_store_root_given(neox, tmp_path):
    vector = torch.randn(D)
    stored = put_tensor(vector, root=tmp_path / "other")
    with pytest.raises(FileNotFoundError):
        load(stored)  # it is not in the default store
    target = ["blocks.1.hook_resid_post"]
    op = Op(op_id="v", type="steer", targets=target, params={"vector": stored})
    inline = Op(op_id="v", type="steer", targets=target, params={"vector": vector.tolist()})
    assert torch.allclose(_logits(neox, _raw(op), root=tmp_path / "other"),
                          _logits(neox, _raw(inline)), atol=1e-6)  # fmt: skip


def test_a_bias_edit_gives_the_edited_biass_logits(neox):
    delta = torch.randn(D)
    op = Op(op_id="b", type="dense_add", targets=["blocks.0.mlp.b_out"],
            params={"delta": put_tensor(delta)})  # fmt: skip
    expected = _edited(neox, lambda m: m.gpt_neox.layers[0].mlp.dense_4h_to_h.bias.add_(delta))
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)


def test_a_sparse_column_edit_of_a_weight_and_of_a_bias(neox):
    cols, values = [2, 5], torch.randn(MLP, 2)
    op = Op(op_id="s", type="sparse_add", targets=["blocks.0.mlp.W_in"],
            params={"axis": "col", "index": cols, "values": put_tensor(values)})  # fmt: skip
    expected = _edited(neox, lambda m: m.gpt_neox.layers[0].mlp.dense_h_to_4h.weight
                       .index_add_(1, torch.tensor(cols), values))  # fmt: skip
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)
    rows, bias = [1, 4], torch.randn(2)
    op = Op(op_id="s", type="sparse_add", targets=["blocks.0.mlp.b_in"],
            params={"axis": "col", "index": rows, "values": put_tensor(bias)})  # fmt: skip
    expected = _edited(neox, lambda m: m.gpt_neox.layers[0].mlp.dense_h_to_4h.bias
                       .index_add_(0, torch.tensor(rows), bias))  # fmt: skip
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)


def _by_hook(model, module, fn):
    """The logits with ``fn(input, output)`` replacing ``module``'s output."""
    handle = module.register_forward_hook(lambda _m, args, out: fn(args[0], out))
    try:
        return logits(model, IDS)
    finally:
        handle.remove()


@pytest.mark.parametrize("extra", [(), ("b_in",), ("b_in", "W_gate")])
def test_added_neurons_add_their_output_to_the_mlps(neox, extra):
    torch.manual_seed(7)
    w = {"W_in": torch.randn(4, D), "W_out": torch.randn(D, 4), "b_in": torch.randn(4),
         "W_gate": torch.randn(4, D)}  # fmt: skip
    keys = ("W_in", "W_out", *extra)
    op = Op(op_id="n", type="add_neurons", targets=["blocks.0.mlp.hook_out"],
            params={"act": "gelu", **{k: put_tensor(w[k]) for k in keys}})  # fmt: skip
    gelu = torch.nn.functional.gelu

    def added(x, out):
        pre = x @ w["W_in"].t() + (w["b_in"] if "b_in" in extra else 0)
        h = gelu(x @ w["W_gate"].t()) * pre if "W_gate" in extra else gelu(pre)
        return out + h @ w["W_out"].t()

    expected = _by_hook(neox, neox.gpt_neox.layers[0].mlp, added)
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)


def _mlp(w, x, act=torch.nn.functional.gelu):
    return act(x @ w["W1"].t() + w["b1"]) @ w["W2"].t() + w["b2"]


@pytest.mark.parametrize("ln", [False, True])
def test_modules_and_added_layers_add_their_block_to_the_residual(neox, ln):
    torch.manual_seed(8)
    w = {"W": torch.randn(D, D), "b": torch.randn(D), "W1": torch.randn(8, D),
         "b1": torch.randn(8), "W2": torch.randn(D, 8), "b2": torch.randn(D)}  # fmt: skip
    stored = {k: put_tensor(v) for k, v in w.items()}
    norm = (lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])) if ln else (lambda x: x)
    layer = neox.gpt_neox.layers[1]
    target = ["blocks.1.hook_resid_post"]
    linear = {"kind": "linear", "W": stored["W"], "b": stored["b"]} | ({"ln": True} if ln else {})
    op = Op(op_id="m", type="add_module", targets=target, params={"module": linear})
    expected = _by_hook(neox, layer, lambda _x, out: out + norm(out) @ w["W"].t() + w["b"])
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)
    mlp = {"kind": "mlp", "act": "relu", **{k: stored[k] for k in ("W1", "b1", "W2", "b2")}}
    op = Op(op_id="m", type="add_module", targets=target,
            params={"module": mlp | ({"ln": True} if ln else {})})  # fmt: skip
    expected = _by_hook(neox, layer, lambda _x, out: out + _mlp(w, norm(out), torch.relu))
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)
    if ln:  # an added layer is always normed, with GELU
        op = Op(op_id="l", type="add_layer", targets=target,
                params={k: stored[k] for k in ("W1", "b1", "W2", "b2")})  # fmt: skip
        expected = _by_hook(neox, layer, lambda _x, out: out + _mlp(w, norm(out)))
        assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)


def test_mean_pruning_with_no_value_is_zero_pruning(neox):
    params = {"unit": "neuron", "index": 5}
    mean = Op(op_id="p", type="prune", targets=["blocks.0.mlp.hook_out"],
              params={**params, "mode": "mean"})  # fmt: skip
    zero = Op(op_id="p", type="prune", targets=["blocks.0.mlp.hook_out"], params=params)
    assert torch.allclose(_run_ops(neox, mean), _run_ops(neox, zero), atol=1e-6)


@pytest.fixture
def qwen():
    """A random two-layer Qwen2: its heads are narrower than the residual, and its MLP is gated.

    So its ``W_O`` is not square, and its MLP names its activation ``act_fn``.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=D, intermediate_size=MLP, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                         vocab_size=50)  # fmt: skip
    return Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize("model", ["neox", "qwen"])
def test_pruning_head_zero_is_zeroing_its_own_slice_of_w_o(model, request):
    model = request.getfixturevalue(model)
    op = Op(op_id="p", type="prune", targets=["blocks.0.attn.hook_z"],
            params={"unit": "head", "index": 0})  # fmt: skip
    path = "model.layers.0.self_attn.o_proj" if hasattr(model, "model") else (
        "gpt_neox.layers.0.attention.dense")  # fmt: skip
    width = model.get_submodule(path).weight.shape[1] // 2  # two heads

    def zero_head(other):
        other.get_submodule(path).weight[:, :width] = 0.0

    assert torch.allclose(_run_ops(model, op), _edited(model, zero_head), atol=1e-5)


def test_added_neurons_use_a_gated_mlps_act_fn(qwen):
    own = Op(op_id="n", type="add_neurons", targets=["blocks.0.mlp.hook_out"],
             params={"W_in": T(4, D), "W_out": T(D, 4)})  # fmt: skip
    silu = own.model_copy(update={"params": {**own.params, "act": "silu"}})
    assert torch.allclose(_run_ops(qwen, own), _run_ops(qwen, silu), atol=1e-6)
    assert not torch.allclose(_run_ops(qwen, own), logits(qwen, IDS), atol=1e-4)


@pytest.mark.parametrize("kind", ["neurons", "module", "steer"])
def test_added_terms_keep_a_bfloat16_models_dtype(neox, kind):
    neox = neox.to(torch.bfloat16)
    with applied(neox, _raw(_layer_one_ops()[kind])):
        out = neox(input_ids=IDS).logits
    assert out.dtype == torch.bfloat16


def test_a_modules_bias_defaults_to_zero(neox):
    # At hook_post no norm follows, so a constant bias would show in the logits.
    w = torch.randn(MLP, MLP)
    op = Op(op_id="m", type="add_module", targets=["blocks.0.mlp.hook_post"],
            params={"module": {"kind": "linear", "W": put_tensor(w)}})  # fmt: skip
    w_out = neox.gpt_neox.layers[0].mlp.dense_4h_to_h
    handle = w_out.register_forward_pre_hook(lambda _m, args: (args[0] + args[0] @ w.t(),))
    try:
        expected = logits(neox, IDS)
    finally:
        handle.remove()
    assert torch.allclose(_run_ops(neox, op), expected, atol=1e-5)
