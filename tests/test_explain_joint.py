"""The lazy greedy plan for one prompt or many (``explain_joint.py``, #7061)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openinterpretability import explain_joint as ej


class Table:
    """A prompt whose raw coverage of a kept set is ``table[frozenset(units)]``, else 0."""

    def __init__(self, table: dict):
        self.table, self.calls = {frozenset(k): v for k, v in table.items()}, []

    def raw(self, sets):
        self.calls.append(len(sets))
        return [self.table.get(frozenset(units), 0.0) for units in sets]


def _exps(*names):
    return [SimpleNamespace(target=n, units=[n]) for n in names]


def test_one_prompt_picks_by_coverage_and_counts_a_pair_that_helps_only_together():
    prompt = Table({"": 0.1, "a": 0.15, "c": 0.4, "bc": 0.7, "ac": 0.42})
    plan = ej.greedy([prompt], _exps("a", "b", "c"), 0.6)
    assert [e.target for e in plan.chosen] == ["c", "b"]
    assert plan.now == [0.1] and plan.picked == [0.7] and plan.reached == 1
    assert plan.gains == pytest.approx([0.3, 0.3])
    assert plan.scans == 1  # one full scan, then the pool was enough


def test_the_pool_holds_the_best_by_the_last_scan_and_a_stall_scans_again():
    # By the first scan, "a" and "b" are best. With "a" kept, neither "b" nor the others help
    # until "c", which the first scan ranked last, is scored again.
    prompt = Table({"a": 0.3, "b": 0.2, "ac": 0.9})
    plan = ej.greedy([prompt], _exps("a", "b", "d", "c"), 0.8, pool=2)
    assert [e.target for e in plan.chosen] == ["a", "c"]
    assert plan.scans == 2
    # Now, the first scan, the pool [a, b], the pool [b, d], the second scan, then its pool.
    assert prompt.calls == [1, 4, 2, 2, 3, 2]


def test_the_plan_stops_when_nothing_helps_and_says_the_target_is_not_reached():
    prompt = Table({"a": 0.2})
    plan = ej.greedy([prompt], _exps("a", "b"), 0.5)
    assert [e.target for e in plan.chosen] == ["a"] and plan.picked == [0.2]
    assert plan.reached == 0 and plan.scans == 2


def test_a_gain_below_epsilon_does_not_count_as_help():
    prompt = Table({"": 0.2, "a": 0.2 + ej.EPSILON / 2})
    assert ej.greedy([prompt], _exps("a"), 0.5).chosen == []
    prompt = Table({"": 0.2, "a": 0.2 + 2 * ej.EPSILON})
    assert [e.target for e in ej.greedy([prompt], _exps("a"), 0.5).chosen] == ["a"]


def test_a_reached_target_needs_no_pick_and_no_scan():
    prompt = Table({"": 0.9})
    plan = ej.greedy([prompt], _exps("a"), 0.8)
    assert plan.chosen == [] and plan.scans == 0 and plan.reached == 1
    assert prompt.calls == [1]  # only the coverage now


def test_a_prompt_within_the_tolerance_has_reached_its_target():
    prompt = Table({"": 0.2, "a": 1.0 - ej.TOLERANCE / 2, "ab": 1.0})
    plan = ej.greedy([prompt], _exps("a", "b"), 1.0)
    assert [e.target for e in plan.chosen] == ["a"] and plan.reached == 1
    prompt = Table({"": 0.2, "a": 1.0 - 2 * ej.TOLERANCE, "ab": 1.0})
    assert [e.target for e in ej.greedy([prompt], _exps("a", "b"), 1.0).chosen] == ["a", "b"]


def test_steps_cap_the_picks():
    prompt = Table({"a": 0.1, "ab": 0.2, "abc": 0.3})
    plan = ej.greedy([prompt], _exps("a", "b", "c"), 1.0, steps=2)
    assert [e.target for e in plan.chosen] == ["a", "b"]


def test_no_candidates_leaves_the_plan_empty():
    plan = ej.greedy([Table({})], [], 0.5)
    assert plan.chosen == [] and plan.scans == 1 and plan.picked == [0.0]


def test_many_prompts_share_a_unit_studied_once():
    # "s" helps both prompts a little. "x" helps only the first prompt, but more.
    first = Table({"s": 0.4, "x": 0.5, "sx": 0.9, "sy": 0.4, "sxy": 0.9})
    second = Table({"s": 0.4, "y": 0.3, "sy": 0.9, "sx": 0.45, "sxy": 0.9})
    plan = ej.greedy([first, second], _exps("x", "s", "y"), 0.8)
    assert [e.target for e in plan.chosen] == ["s", "x", "y"]
    assert plan.picked == pytest.approx([0.9, 0.9]) and plan.reached == 2
    assert plan.gains[0] == pytest.approx(0.8)  # summed over both prompts


def test_a_prompt_past_the_target_adds_nothing_to_the_value():
    # "a" raises the first prompt past the target, "b" lifts the second, which needs it.
    first = Table({"": 0.9, "a": 1.0, "b": 0.9, "ab": 1.0})
    second = Table({"": 0.0, "b": 0.05, "ab": 0.05})
    plan = ej.greedy([first, second], _exps("a", "b"), 0.8)
    assert [e.target for e in plan.chosen] == ["b"]


def test_capped_coverage_is_at_most_one_and_minus_one_when_missing():
    plan = ej.greedy([Table({"": 1.4})], _exps("a"), 1.0)
    assert plan.now == [1.0]
    missing = SimpleNamespace(raw=lambda sets: [None] * len(sets))
    assert ej.greedy([missing], _exps("a"), 0.5).now == [-1.0]


def test_candidates_of_several_plans_keep_each_target_once_in_first_seen_order():
    a, b, c = _exps("a", "b", "c")
    other_b = SimpleNamespace(target="b", units=["b"])
    found = ej.candidates_of([SimpleNamespace(experiments=[b, a]),
                              SimpleNamespace(experiments=[other_b, c])])  # fmt: skip
    assert found == [b, a, c] and found[0] is b
