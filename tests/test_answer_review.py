"""Answer review checks before any measure, and the units of a model without heads."""

from types import SimpleNamespace

import pytest
import torch

from openinterpretability.answer_review import answer_token, check_claim, every_unit
from tests.conftest import FakeTokenizer
from tests.test_answer_records import _claim


def test_an_answer_of_two_tokens_has_no_token_id():
    runner = SimpleNamespace(tokenizer=FakeTokenizer())
    with pytest.raises(ValueError, match="the answer 'ef' is 2 tokens, not one"):
        answer_token(runner, "ef")
    assert answer_token(runner, "e") == FakeTokenizer().encode("e")[0]


def test_a_claim_with_a_long_answer_and_a_bad_label_lists_both_problems():
    problems = check_claim(FakeTokenizer(), _claim(answer="ef", route=["L0/H1", "L1/F3"]))
    assert problems[0] == "the answer 'ef' is not one token"
    assert len(problems) == 2 and "not a unit label: 'L1/F3'" in problems[1]


def test_every_unit_without_head_means_is_only_neuron_poles():
    means = SimpleNamespace(neurons={0: torch.zeros(2)}, heads={})
    units = every_unit(SimpleNamespace(n_heads=4), means)
    assert units == [(0, 0, "positive"), (0, 0, "negative"), (0, 1, "positive"), (0, 1, "negative")]
