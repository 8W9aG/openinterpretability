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


def gate_op(op_id, gates=None, target=W_OUT):
    """A gate op: a token match, or a composite of ``gates``."""
    function = {"kind": "or", "gates": gates} if gates else {"kind": "token_match", "tokens": [5]}
    return Op(op_id=op_id, type="gate", targets=[target], params={"function": function})


def test_a_diamond_lists_each_ancestor_once():
    root = patch([])
    left, right = patch([dense("l")], parents=[root.id]), patch([dense("r")], parents=[root.id])
    tip = patch([], parents=[left.id, right.id])
    order = history.ancestors(tip.id, lookup_of(root, left, right, tip))
    assert [p.id for p in order] == [root.id, left.id, right.id, tip.id]
    assert [p.id for p in history.log(tip.id, lookup_of(root, left, right, tip))] == [
        tip.id, right.id, left.id, root.id]  # fmt: skip


def test_replay_writes_gates_as_full_refs_and_keeps_refs_that_are_full():
    root = patch([gate_op("g")])
    child = patch([gate_op("both", gates=["own", ref(root.id, "g")]), gate_op("own"),
                   dense("d").model_copy(update={"gate": "own"})], parents=[root.id])  # fmt: skip
    ops = dict(history.replay(child.id, lookup_of(root, child)))
    assert ops[ref(child.id, "both")].params["function"]["gates"] == [
        ref(child.id, "own"), ref(root.id, "g")]  # fmt: skip
    assert ops[ref(child.id, "d")].gate == ref(child.id, "own")
    assert ops[ref(root.id, "g")].gate is None


def test_a_set_op_retunes_only_its_targets_and_merges_nested_params():
    first = Op(op_id="a", type="steer", targets=[W_OUT],
               params={"vector": "v", "opts": {"x": 1, "y": 2}, "n": 5})  # fmt: skip
    second = Op(op_id="b", type="steer", targets=[W_OUT], params={"vector": "w"})
    root = patch([first, second])
    setter = Op(op_id="s", type="set", targets=[ref(root.id, "a")],
                params={"params": {"opts": {"y": 3}, "n": {"k": 1}, "new": {"z": 0}}})  # fmt: skip
    child = patch([setter], parents=[root.id])
    ops = dict(history.replay(child.id, lookup_of(root, child)))
    assert ops[ref(root.id, "a")].params == {"vector": "v", "opts": {"x": 1, "y": 3},
                                             "n": {"k": 1}, "new": {"z": 0}}  # fmt: skip
    assert ops[ref(root.id, "b")].params == {"vector": "w"}


def test_a_set_op_must_name_ops_the_history_holds():
    root = patch([dense("a")])
    setter = Op(op_id="s", type="set", targets=[ref(root.id, "gone")], params={"params": {}})
    child = patch([setter], parents=[root.id])
    with pytest.raises(PatchError, match=r"^set op 's' names unknown ops \["):
        history.replay(child.id, lookup_of(root, child))


def test_the_order_error_is_the_whole_message():
    root = patch([dense("a")])
    child = patch([Op(op_id="o", type="order", targets=[ref(root.id, "gone")])], parents=[root.id])
    with pytest.raises(PatchError, match="^an order op names refs that the history doesn't hold$"):
        history.replay(child.id, lookup_of(root, child))


def test_diff_lists_ops_on_one_side_and_ops_with_other_params():
    root = patch([dense("a"), dense("b")])
    retune = Op(op_id="s", type="set", targets=[ref(root.id, "a")],
                params={"params": {"delta": "u"}})  # fmt: skip
    left = patch([retune, dense("l")], parents=[root.id])
    right = patch([dense("r")], parents=[root.id])
    found = history.diff(left.id, right.id, lookup_of(root, left, right))
    assert found == {"only_a": [ref(left.id, "l")], "only_b": [ref(right.id, "r")],
                     "changed": [ref(root.id, "a")]}  # fmt: skip


@pytest.mark.parametrize("dense_first", [True, False])
def test_a_dense_edit_with_an_axis_still_overlaps_a_sparse_edit(dense_first):
    root = patch([])
    whole = Op(op_id="d", type="dense_add", targets=[W_OUT], params={"delta": "t", "axis": "row"})
    ops = [whole, sparse("s", [4])]
    left, right = (patch([op], parents=[root.id]) for op in (ops if dense_first else ops[::-1]))
    (found,) = history.conflicts(left.id, right.id, lookup_of(root, left, right))
    assert found.kind == "parameter"


def test_a_gate_on_the_right_never_conflicts():
    root = patch([])
    left, right = patch([dense("d")], parents=[root.id]), patch([gate_op("g")], parents=[root.id])
    assert history.conflicts(left.id, right.id, lookup_of(root, left, right)) == []


def test_a_weight_edit_and_an_additive_activation_edit_commute():
    root = patch([])
    steer = Op(op_id="v", type="steer", targets=[W_OUT], params={"vector": "v"})
    left, right = patch([dense("d")], parents=[root.id]), patch([steer], parents=[root.id])
    assert history.conflicts(left.id, right.id, lookup_of(root, left, right)) == []


def test_an_additive_and_a_non_additive_edit_need_an_order():
    root = patch([])
    clamp = Op(op_id="c", type="clamp", targets=[W_OUT], params={"index": 1, "value": 0})
    steer = Op(op_id="v", type="steer", targets=[W_OUT], params={"vector": "v"})
    left, right = patch([steer], parents=[root.id]), patch([clamp], parents=[root.id])
    found = history.conflicts(left.id, right.id, lookup_of(root, left, right))
    assert found == [history.Conflict("order", W_OUT, (ref(left.id, "v"), ref(right.id, "c")))]


def _footprinted():
    """A root and two branches whose footprints share one unit beyond the root's."""
    root = Patch(base=BASE, evidence={"footprint": ["L1/N5"]}).sealed()
    left = Patch(base=BASE, parents=[root.id], ops=[dense("l", target="a")],
                 evidence={"footprint": ["L1/N3", "L1/N5"]}).sealed()  # fmt: skip
    right = Patch(base=BASE, parents=[root.id], ops=[dense("r", target="b")],
                  evidence={"footprint": ["L1/N3", "L1/N9"]}).sealed()  # fmt: skip
    return root, left, right


def test_units_in_both_footprints_beyond_the_shared_history_conflict():
    root, left, right = _footprinted()
    found = history.conflicts(left.id, right.id, lookup_of(root, left, right))
    assert found == [history.Conflict("footprint", "L1/N3", (left.id, right.id))]


def test_a_merge_records_footprint_conflicts_and_keeps_its_provenance():
    root, left, right = _footprinted()
    merged = history.merge([left.id, right.id], [], {"why": "x"}, lookup_of(root, left, right),
                           provenance={"by": "test"})  # fmt: skip
    assert merged.base == BASE and merged.parents == [left.id, right.id]
    assert merged.message == {"why": "x",
                              "footprint_conflicts": [{"where": "L1/N3",
                                                       "refs": [left.id, right.id]}]}  # fmt: skip
    assert merged.provenance == {"by": "test"} and merged.id == merged.content_id()


def test_a_merge_with_no_footprint_conflict_keeps_its_message():
    root = patch([])
    left, right = (
        patch([dense("l", "a")], parents=[root.id]),
        patch([dense("r", "b")], parents=[root.id]),
    )
    merged = history.merge([left.id, right.id], [], {"why": "x"}, lookup_of(root, left, right))
    assert merged.message == {"why": "x"} and merged.provenance == {}


def test_a_merge_of_parents_on_other_bases_is_refused():
    left, right = patch([]), patch([], base="other/model@1")
    with pytest.raises(PatchError, match=r"^a merge's parents must share one base, not \["):
        history.merge([left.id, right.id], [], {}, lookup_of(left, right))


def test_a_merge_needs_two_parents_says_so_exactly():
    root = patch([])
    with pytest.raises(PatchError, match="^a merge has two or more parents$"):
        history.merge([root.id], [], {}, lookup_of(root))


def test_a_revert_drops_the_refs_and_keeps_its_provenance():
    root = patch([dense("a"), dense("b")])
    undo = history.revert(root.id, [ref(root.id, "a")], {"why": "x"}, lookup_of(root),
                          provenance={"by": "test"})  # fmt: skip
    assert undo.ops == [Op(op_id="revert", type="drop", targets=[ref(root.id, "a")])]
    assert (undo.base, undo.parents, undo.message) == (BASE, [root.id], {"why": "x"})
    assert undo.provenance == {"by": "test"}
    assert [r for r, _ in history.replay(undo.id, lookup_of(root, undo))] == [ref(root.id, "b")]


def test_a_rebase_copies_the_chain_onto_the_new_parent_and_rewrites_its_refs():
    onto = patch([gate_op("g")], base="m@2")
    old_root = patch([])
    first = Patch(base=BASE, parents=[old_root.id], ops=[gate_op("own")], message={"m": 1},
                  provenance={"by": "test"}, evidence={"verdict": "fixed"}).sealed()  # fmt: skip
    outside = ref(onto.id, "g")  # a ref to a patch outside the chain stays as it is
    second = patch([dense("d").model_copy(update={"gate": ref(first.id, "own")}),
                    dense("e", target=outside).model_copy(update={"gate": outside}),
                    dense("f")], parents=[first.id])  # fmt: skip
    new_first, new_second = history.rebase([first.id, second.id], onto.id, "m@2",
                                           lookup_of(onto, old_root, first, second))  # fmt: skip
    assert (new_first.base, new_first.parents, new_first.message) == ("m@2", [onto.id], {"m": 1})
    assert new_first.provenance == {"by": "test"} and new_first.evidence == {}
    assert new_first.rebased_from == first.id and new_first.id == new_first.content_id()
    d, e, f = new_second.ops
    assert d.gate == ref(new_first.id, "own")
    assert (e.gate, e.targets) == (ref(onto.id, "g"), [ref(onto.id, "g")])
    assert (f.gate, f.targets) == (None, [W_OUT])
    assert new_second.parents == [new_first.id]


def test_a_rebase_onto_the_base_has_no_parent():
    first = patch([dense("a")])
    (copy,) = history.rebase([first.id], None, "m@2", lookup_of(first))
    assert copy.parents == [] and copy.base == "m@2"
