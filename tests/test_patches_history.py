"""Patch history: the errors a bad history raises, and which branch ops conflict (#5229)."""

from __future__ import annotations

import pytest

from openinterpretability.patches import history
from openinterpretability.patches.schema import Op, Patch, PatchError, ref

BASE = "EleutherAI/pythia-70m@abc123"
W_OUT = "blocks.1.mlp.W_out"


def patch(ops, parents=(), base=BASE) -> Patch:
    return Patch(base=base, parents=list(parents), ops=ops).sealed()


def lookup_of(*patches):
    return {p.id: p for p in patches}.__getitem__


def sparse(op_id, rows, target=W_OUT):
    return Op(op_id=op_id, type="sparse_add", targets=[target],
              params={"axis": "row", "index": rows, "values": "t"})  # fmt: skip


def dense(op_id, target=W_OUT):
    return Op(op_id=op_id, type="dense_add", targets=[target], params={"delta": "t"})


def test_a_patch_that_is_its_own_ancestor_is_refused():
    loop = Patch(base=BASE, parents=["a" * 64])
    with pytest.raises(PatchError, match=f"patch {'a' * 12} is its own ancestor"):
        history.ancestors("a" * 64, {"a" * 64: loop}.__getitem__)


def test_a_chain_that_mixes_bases_is_refused():
    root = patch([], base="other/model@1")
    child = patch([], parents=[root.id])
    with pytest.raises(PatchError, match=r"a patch chain mixes bases: \['Eleuther"):
        history.ancestors(child.id, lookup_of(root, child))


def test_an_order_op_must_name_refs_the_history_holds():
    root = patch([sparse("a", [0])])
    order = Op(op_id="o", type="order", targets=[ref(root.id, "a"), ref(root.id, "gone")])
    child = patch([order], parents=[root.id])
    with pytest.raises(PatchError, match="an order op names refs that the history doesn't hold"):
        history.replay(child.id, lookup_of(root, child))


def test_a_drop_op_must_name_ops_the_history_holds():
    root = patch([sparse("a", [0])])
    drop = Op(op_id="d", type="drop", targets=[ref(root.id, "gone")])
    child = patch([drop], parents=[root.id])
    with pytest.raises(PatchError, match=r"drop op 'd' names unknown ops \["):
        history.replay(child.id, lookup_of(root, child))


def test_a_dense_edit_conflicts_with_a_sparse_edit_of_the_same_parameter():
    root = patch([])
    left = patch([dense("d")], parents=[root.id])
    right = patch([sparse("s", [4])], parents=[root.id])
    (found,) = history.conflicts(left.id, right.id, lookup_of(root, left, right))
    assert found == history.Conflict("parameter", W_OUT, (ref(left.id, "d"), ref(right.id, "s")))


def test_ops_on_other_targets_and_gates_never_conflict():
    root = patch([])
    gate = Op(op_id="g", type="gate", targets=[W_OUT],
              params={"function": {"kind": "token_match", "tokens": [5]}})  # fmt: skip
    left = patch([sparse("a", [0]), gate], parents=[root.id])
    right = patch([sparse("b", [0], target="blocks.0.mlp.W_in"), dense("c")], parents=[root.id])
    found = history.conflicts(left.id, right.id, lookup_of(root, left, right))
    # Only the two weight edits of W_OUT conflict: the gate and the W_in edit are skipped.
    assert found == [history.Conflict("parameter", W_OUT, (ref(left.id, "a"), ref(right.id, "c")))]


def test_a_merge_needs_two_parents():
    root = patch([])
    with pytest.raises(PatchError, match="a merge has two or more parents"):
        history.merge([root.id], [], {}, lookup_of(root))


def test_a_revert_of_an_op_that_is_not_applied_is_refused():
    root = patch([sparse("a", [0])])
    gone = ref(root.id, "gone")
    with pytest.raises(PatchError, match="nothing to revert at"):
        history.revert(root.id, [ref(root.id, "a"), gone], {}, lookup_of(root))
