"""Fix patches: format, apply, history and branches (openinterpretability/patches, #5229)."""

from __future__ import annotations

import pytest
import torch

from openinterpretability.patches import history
from openinterpretability.patches.apply import applied, logits, zero_start_check
from openinterpretability.patches.schema import Op, Patch, PatchError, read_patch, ref
from openinterpretability.patches.store import PatchStore
from openinterpretability.patches.tensors import put_tensor

BASE = "EleutherAI/pythia-70m@abc123"
IDS = torch.tensor([[3, 7, 11, 7, 2, 9]])
D, MLP, VOCAB = 32, 64, 50


@pytest.fixture(autouse=True)
def tensor_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_EVIDENCE_DIR", str(tmp_path / "evidence"))


@pytest.fixture
def neox():
    """A random two-layer GPT-NeoX built from config: Pythia's architecture, no download."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    sizes = {"hidden_size": D, "intermediate_size": MLP, "vocab_size": VOCAB}
    config = GPTNeoXConfig(num_hidden_layers=2, num_attention_heads=2, **sizes)
    return GPTNeoXForCausalLM(config).eval()


def T(*shape, zero=False):
    return put_tensor(torch.zeros(*shape) if zero else torch.randn(*shape))


def patch(ops, parents=(), **extra) -> Patch:
    return Patch(base=BASE, parents=list(parents), ops=ops, **extra).sealed()


def lookup_of(*patches):
    table = {p.id: p for p in patches}
    return table.__getitem__


def run(model, chain, lookup, strength=1.0):
    with applied(model, history.replay(chain, lookup), strength):
        return logits(model, IDS)


def sparse(op_id, rows, target="blocks.1.mlp.W_out"):
    values = torch.randn(len(rows), MLP)
    return Op(op_id=op_id, type="sparse_add", targets=[target],
              params={"axis": "row", "index": rows, "values": put_tensor(values)})  # fmt: skip


def test_a_zero_scale_patch_gives_the_base_logits(neox):
    ops = [Op(op_id="lr", type="lowrank_add", targets=["blocks.0.mlp.W_in"],
              params={"A": T(4, D), "B": T(MLP, 4)}),
           Op(op_id="steer", type="steer", targets=["blocks.1.hook_resid_post"],
              params={"vector": T(D)})]  # fmt: skip
    p = patch(ops)
    replayed = history.replay(p.id, lookup_of(p))
    assert zero_start_check(neox, replayed, IDS) == 0.0
    assert not torch.equal(run(neox, p.id, lookup_of(p)), logits(neox, IDS))  # it does act


def test_a_zero_scale_patch_on_pythia_70m(tmp_path):
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", local_files_only=True)
    except OSError:
        pytest.skip("pythia-70m is not in the local Hugging Face cache")
    delta = torch.randn(model.config.hidden_size, model.config.intermediate_size)
    op = Op(op_id="add", type="dense_add", targets=["blocks.3.mlp.W_out"],
            params={"delta": put_tensor(delta)})  # fmt: skip
    p = patch([op])
    zero_start_check(model.eval(), history.replay(p.id, lookup_of(p)), IDS)


def test_a_two_patch_chain_round_trips_with_its_hashes(neox, tmp_path):
    store = PatchStore(tmp_path / "wiki", BASE)
    root = patch([sparse("a", [0, 1])])
    child = patch([sparse("b", [5])], parents=[root.id])
    for p in (root, child):
        store.write(p, sign=False)
    assert store.get(child.id[:10]) == child  # a prefix is enough
    before = logits(neox, IDS)
    patched = run(neox, child.id, store.get)
    assert not torch.allclose(patched, before)
    assert torch.equal(logits(neox, IDS), before)  # removed exactly
    text = store.path(child.id).read_text().replace('"b"', '"c"')
    with pytest.raises(PatchError, match="content hash"):
        read_patch(text)


def test_branches_merge_and_a_parameter_conflict_blocks_until_resolved(neox, tmp_path):
    root = patch([])
    left = patch([sparse("l", [0, 1])], parents=[root.id])
    right = patch([sparse("r", [1, 2])], parents=[root.id])  # row 1 overlaps
    clean = patch([sparse("c", [7])], parents=[root.id])
    lookup = lookup_of(root, left, right, clean)
    assert [c.kind for c in history.conflicts(left.id, right.id, lookup)] == ["parameter"]
    with pytest.raises(PatchError, match="parameter"):
        history.merge([left.id, right.id], [], {"text": "both"}, lookup)
    resolved = history.merge([left.id, right.id], [], {"resolution": "additive"}, lookup)
    assert resolved.parents == [left.id, right.id]
    merged = history.merge([left.id, clean.id], [], {"text": "clean"}, lookup)
    both = lookup_of(root, left, right, clean, merged)
    assert [r for r, _ in history.replay(merged.id, both)] == [
        ref(left.id, "l"),
        ref(clean.id, "c"),
    ]
    one = run(neox, left.id, both)
    assert not torch.allclose(run(neox, merged.id, both), one)  # both deltas apply


def test_main_moves_only_to_a_fixed_patch(tmp_path):
    store = PatchStore(tmp_path / "wiki", BASE)
    trial = patch([sparse("a", [0])], evidence={"verdict": "not fixed"})
    good = patch([sparse("a", [1])], evidence={"verdict": "fixed"})
    for p in (trial, good):
        store.write(p, sign=False)
    store.move("fix/copy-unit", trial.id, create=True)
    with pytest.raises(PatchError, match="fixed"):
        store.move("main", trial.id, create=True)
    store.move("main", good.id, create=True)
    assert store.branches() == {"fix/copy-unit": trial.id, "main": good.id}
    store.delete("fix/copy-unit")
    assert store.path(trial.id).exists()  # the patch stays


def test_a_revert_restores_the_parents_logits(neox):
    parent = patch([sparse("a", [0, 3])])
    child = patch([Op(op_id="clamp", type="clamp", targets=["blocks.0.mlp.hook_post"],
                      params={"index": 5, "value": 4.0})], parents=[parent.id])  # fmt: skip
    undo = history.revert(child.id, [ref(child.id, "clamp")], {"why": "test"},
                          lookup_of(parent, child))  # fmt: skip
    lookup = lookup_of(parent, child, undo)
    expected = run(neox, parent.id, lookup)
    assert not torch.allclose(run(neox, child.id, lookup), expected)
    assert torch.allclose(run(neox, undo.id, lookup), expected, atol=1e-5)


def _gated_module(force=None):
    probe = Op(op_id="probe", type="gate", params={
        "function": {"kind": "linear_probe", "input": "blocks.0.hook_resid_post",
                     "w": T(D), "b": 0.0}, "output": "soft", "threshold": -10.0})  # fmt: skip
    attention = Op(op_id="attn", type="gate", params={
        "function": {"kind": "attention_score", "heads": [[0, 1], [1, 0]], "key_tokens": [7]},
        "output": "hard", "threshold": -1.0})  # fmt: skip
    both = Op(op_id="both", type="gate", params={
        "function": {"kind": "and", "gates": ["probe", "attn"]}, "positions": "last",
        **({"force": force} if force else {}),
        "meaning": "opens when the last position attends to token 7"})  # fmt: skip
    module = Op(op_id="fix", type="add_module", targets=["blocks.1.hook_resid_post"], gate="both",
                params={"module": {"kind": "mlp", "W1": T(8, D), "W2": T(D, 8)}})  # fmt: skip
    return patch([probe, attention, both, module])


def test_a_gated_module_round_trips_shuts_and_takes_a_retuned_threshold(neox, tmp_path):
    attention = neox.config._attn_implementation
    store = PatchStore(tmp_path / "wiki", BASE)
    p = _gated_module()
    store.write(p, sign=False)
    assert store.get(p.id) == p
    base = logits(neox, IDS)
    opened = run(neox, p.id, store.get)
    assert not torch.allclose(opened[:, -1], base[:, -1])  # the gate opens at the last position
    assert torch.allclose(opened[:, :-1], base[:, :-1], atol=1e-6)  # and nowhere else
    shut = patch([Op(op_id="shut", type="set", targets=[ref(p.id, "both")],
                     params={"params": {"force": "shut"}})], parents=[p.id])  # fmt: skip
    store.write(shut, sign=False)
    assert torch.allclose(run(neox, shut.id, store.get), base, atol=1e-6)
    retune = patch([Op(op_id="t", type="set", targets=[ref(p.id, "attn")],
                       params={"params": {"threshold": 2.0}})], parents=[p.id])  # fmt: skip
    store.write(retune, sign=False)
    (gate,) = [op for r, op in history.replay(retune.id, store.get) if r == ref(p.id, "attn")]
    assert gate.params["threshold"] == 2.0  # mean attention never exceeds 1: always shut
    assert torch.allclose(run(neox, retune.id, store.get), base, atol=1e-6)
    assert neox.config._attn_implementation == attention  # restored after each run


@pytest.mark.parametrize("kind", ["add_neurons", "add_layer"])
def test_structural_ops_pass_the_zero_start_check(neox, kind):
    if kind == "add_neurons":
        op = Op(op_id="n", type=kind, targets=["blocks.0.mlp.hook_out"], identity_at_init=True,
                params={"W_in": T(4, D), "b_in": T(4), "W_out": T(D, 4, zero=True)})  # fmt: skip
    else:
        op = Op(op_id="l", type=kind, targets=["blocks.0.hook_resid_post"], identity_at_init=True,
                params={"W1": T(16, D), "W2": T(D, 16, zero=True)})  # fmt: skip
    p = patch([op])
    replayed = history.replay(p.id, lookup_of(p))
    zero_start_check(neox, replayed, IDS)  # every output scale at zero
    zero_start_check(neox, replayed, IDS, strength=1.0)  # identity as it starts


def test_the_reader_refuses_unknown_types_and_newer_registries():
    good = patch([sparse("a", [0])])
    unknown = good.model_dump_json().replace('"sparse_add"', '"teleport"')
    with pytest.raises(PatchError, match="unknown operation type"):
        read_patch(unknown)
    newer = good.model_dump_json().replace('"registry_version":1', '"registry_version":99')
    with pytest.raises(PatchError, match="registry 99"):
        read_patch(newer)


def test_two_non_additive_ops_at_one_hook_point_are_an_order_conflict():
    root = patch([])
    clamp = {"index": 2, "value": 1.0}
    left = patch([Op(op_id="x", type="clamp", targets=["blocks.0.mlp.hook_post"], params=clamp)],
                 parents=[root.id])  # fmt: skip
    right = patch([Op(op_id="y", type="clamp", targets=["blocks.0.mlp.hook_post"],
                      params={**clamp, "value": 3.0})], parents=[root.id])  # fmt: skip
    lookup = lookup_of(root, left, right)
    (found,) = history.conflicts(left.id, right.id, lookup)
    assert found.kind == "order" and found.where == "blocks.0.mlp.hook_post"
    with pytest.raises(PatchError, match="order"):
        history.merge([left.id, right.id], [], {}, lookup)
    order = Op(op_id="o", type="order", targets=[ref(right.id, "y"), ref(left.id, "x")])
    merged = history.merge([left.id, right.id], [order], {}, lookup)
    replayed = history.replay(merged.id, lookup_of(root, left, right, merged))
    assert [r for r, _ in replayed] == [ref(right.id, "y"), ref(left.id, "x")]


def test_a_rebase_links_its_originals_and_drops_their_evidence():
    root = patch([sparse("a", [0])], evidence={"verdict": "fixed"})
    child = patch([Op(op_id="s", type="set", targets=[ref(root.id, "a")],
                      params={"params": {"scale": 0.5}})], parents=[root.id])  # fmt: skip
    new = history.rebase([root.id, child.id], None, "EleutherAI/pythia-70m@def456",
                         lookup_of(root, child))  # fmt: skip
    assert [p.rebased_from for p in new] == [root.id, child.id]
    assert new[0].evidence == {} and new[1].parents == [new[0].id]
    assert new[1].ops[0].targets == [ref(new[0].id, "a")]
    assert history.replay(new[1].id, lookup_of(*new))[0][1].params["scale"] == 0.5
