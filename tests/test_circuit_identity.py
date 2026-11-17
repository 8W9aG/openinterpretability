"""``scripts/circuit_identity.py``: circuit ids across builds (#4283)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import circuit_identity as ci  # noqa: E402


def test_jaccard_is_one_minus_scaled_set_edit_distance():
    a, b = frozenset("abcd"), frozenset("abce")
    assert ci.jaccard(a, b) == 3 / 5 and ci.jaccard(a, a) == 1.0


def test_a_circuit_keeps_its_id_when_it_grows():
    old = {"C1": frozenset("abcd")}
    named, events, next_id = ci.step(old, [frozenset("abcde")], 2)
    assert named == {"C1": frozenset("abcde")} and events["grow_or_shrink"] == 1
    assert next_id == 2


def test_a_cluster_that_overlaps_too_little_is_new_and_the_old_one_is_gone():
    old = {"C1": frozenset("abcd")}
    named, events, next_id = ci.step(old, [frozenset("defg")], 2)
    assert named == {"C2": frozenset("defg")} and events["new"] == 1 and events["gone"] == 1


def test_splits_and_merges_are_counted():
    _, split, _ = ci.step({"C1": frozenset("abcdef")}, [frozenset("abc"), frozenset("def")], 2)
    assert split["split"] == 1
    old = {"C1": frozenset("abc"), "C2": frozenset("def")}
    _, merge, _ = ci.step(old, [frozenset("abcdef")], 3)
    assert merge["merge"] == 1


def test_the_matching_pairs_each_old_circuit_once():
    old = {"C1": frozenset("abcd"), "C2": frozenset("wxyz")}
    pairs = ci.match(old, [frozenset("wxyz"), frozenset("abce")])
    assert {i: cid for i, (cid, _) in pairs.items()} == {0: "C2", 1: "C1"}
