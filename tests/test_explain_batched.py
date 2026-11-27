"""The batched coverage measure (``openinterpretability/explain_batched.py``, #7061)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from openinterpretability import explain as ex
from openinterpretability import explain_coverage as ec
from openinterpretability.explain_batched import KeptSets
from openinterpretability.explain_groups import HEAD
from tests.conftest import FakeTokenizer

TEXTS = ["the cat sat on the mat", "a dog ran in the park and sat", "hello there friend"] * 3
PROMPT = "the cat sat on"


@pytest.fixture
def gpt2(tiny_gpt2, tmp_path, monkeypatch):
    from openinterpretability.harness.hooks import NeuronHookRunner

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    return runner, ex.unit_means(runner, "tiny", TEXTS, 200)


def _reference(runner, means, units, token, explained=()):
    """The raw coverage of ``units`` with one forward pass per set, as keep_hooks gives it."""
    clean = ec._log_prob(runner, PROMPT, token, -1)
    ablated = ec._log_prob(runner, PROMPT, token, -1, ec.keep_hooks(runner, means, []))
    kept = ec._log_prob(runner, PROMPT, token, -1,
                        ec.keep_hooks(runner, means, [*explained, *units]))  # fmt: skip
    return ec._raw(clean, ablated, kept)


SETS = [[], [(0, 3, "positive")], [(1, 0, HEAD)], [(0, 3, "positive"), (0, 3, "negative")],
        [(0, 1, "negative"), (1, 5, "positive"), (0, 1, HEAD), (1, 1, HEAD)]]  # fmt: skip


def test_each_set_gets_what_keep_hooks_gives_it_one_pass_at_a_time(gpt2):
    runner, means = gpt2
    assert means.heads  # the tiny GPT-2 has heads, so both hooks are checked
    token = 5
    sets = KeptSets(runner, means, PROMPT, token, -1, explained=[], batch=2)  # three batches
    found = sets.raw(SETS)
    expected = [_reference(runner, means, s, token) for s in SETS]
    assert found == pytest.approx(expected, abs=1e-4)
    assert found[0] == pytest.approx(0.0, abs=1e-6)  # nothing kept is the ablated run
    assert len(set(round(v, 4) for v in found)) > 2  # the sets differ, so the test can tell


def test_the_explained_units_are_kept_in_every_set(gpt2):
    runner, means = gpt2
    explained = [(1, 0, HEAD), (0, 2, "positive")]
    sets = KeptSets(runner, means, PROMPT, 7, -1, explained=explained)
    assert sets.raw(SETS[1:3]) == pytest.approx(
        [_reference(runner, means, s, 7, explained) for s in SETS[1:3]], abs=1e-4
    )


def test_the_clean_and_ablated_runs_match_the_one_pass_measure(gpt2):
    runner, means = gpt2
    sets = KeptSets(runner, means, PROMPT, 3, -1, explained=[])
    assert sets.clean == pytest.approx(ec._log_prob(runner, PROMPT, 3, -1), abs=1e-5)
    ablated = ec._log_prob(runner, PROMPT, 3, -1, ec.keep_hooks(runner, means, []))
    assert sets.ablated == pytest.approx(ablated, abs=1e-5)


def test_a_model_with_no_heads_is_scored_on_its_neurons(toy_runner, tmp_path, monkeypatch):
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    means = ex.unit_means(toy_runner, "toy", TEXTS, 200)
    assert not means.heads
    units = [[], [(0, 1, "positive")], [(1, 2, "negative"), (2, 0, "positive")]]
    found = KeptSets(toy_runner, means, PROMPT, 4, -1, explained=[]).raw(units)
    assert found == pytest.approx([_reference_toy(toy_runner, means, u) for u in units], abs=1e-4)


def _reference_toy(runner, means, units):
    return _reference(runner, means, units, 4)


def test_no_coverage_when_ablating_every_unit_changes_nothing(monkeypatch):
    sets = KeptSets.__new__(KeptSets)  # a measure whose clean and ablated runs are equal
    sets.clean = sets.ablated = -2.0
    sets.explained = []
    assert sets.raw([[], [(0, 1, "positive")]]) == [None, None]
    sets.ablated = -2.0 - 1e-10  # still within the tolerance of 1e-9
    assert sets.raw([[]]) == [None]


def test_a_span_just_outside_the_tolerance_gives_a_value(monkeypatch):
    sets = KeptSets.__new__(KeptSets)
    sets.clean, sets.ablated, sets.explained = -2.0, -2.0 - 1e-8, []
    empty = lambda found: (torch.zeros(1, len(found), 1),)  # noqa: E731
    monkeypatch.setattr(sets, "masks", empty, raising=False)
    monkeypatch.setattr(sets, "_log_probs", lambda masks: [-2.0] * masks[0].shape[1], raising=False)
    assert sets.raw([[]]) == [pytest.approx(1.0)]
    assert sets.raw([]) == []
    # raw_masks has its own check, for masks that a caller built.
    sets.ablated = sets.clean
    assert sets.raw_masks((torch.zeros(1, 3, 1),)) == [None, None, None]


def test_the_masks_mark_each_kind_of_unit_in_its_own_row():
    sets = KeptSets.__new__(KeptSets)
    runner = SimpleNamespace(n_layers=2, n_heads=3, device=torch.device("cpu"))
    sets.runner, sets.width, sets.means = runner, 4, SimpleNamespace(heads={0: None})
    sets.explained = []
    rows = [[(0, 1, "positive")], [(1, 2, "negative"), (1, 0, HEAD)]]
    positive, negative, heads = sets.masks(rows)
    assert positive.shape == (2, 2, 4) and heads.shape == (2, 2, 3)
    assert positive.nonzero().tolist() == [[0, 0, 1]]
    assert negative.nonzero().tolist() == [[1, 1, 2]]
    assert heads.nonzero().tolist() == [[1, 1, 0]]


def test_the_explained_units_are_in_every_row_of_the_masks_and_without_heads_one_column():
    sets = KeptSets.__new__(KeptSets)
    runner = SimpleNamespace(n_layers=1, n_heads=5, device=torch.device("cpu"))
    sets.runner, sets.width, sets.means = runner, 3, SimpleNamespace(heads={})
    sets.explained = [(0, 2, "positive")]
    positive, _, heads = sets.masks([[], [(0, 0, "positive")]])
    assert positive.nonzero().tolist() == [[0, 0, 2], [0, 1, 0], [0, 1, 2]]
    assert heads.shape == (1, 2, 1)  # no heads to keep, so n_heads is never read


def test_edited_masks_drop_a_unit_from_one_row(gpt2):
    runner, means = gpt2
    sets = KeptSets(runner, means, PROMPT, 5, -1, explained=[])
    both = [(0, 3, "positive"), (1, 0, HEAD)]
    masks = sets.masks([both, both])
    masks[2][1, 1, 0] = False  # row 1 drops the head
    found = sets.raw_masks(masks)
    assert found == pytest.approx(sets.raw([both, both[:1]]), abs=1e-5)
    assert found[0] != pytest.approx(found[1], abs=1e-4)


def test_one_batch_or_many_give_the_same_values(gpt2):
    runner, means = gpt2
    one = KeptSets(runner, means, PROMPT, 5, -1, explained=[], batch=len(SETS))
    many = KeptSets(runner, means, PROMPT, 5, -1, explained=[], batch=1)
    assert one.raw(SETS) == pytest.approx(many.raw(SETS), abs=1e-4)  # float32 rounding by batch
