"""Word strengths in ``harness.specificity`` when tokens and strengths do not line up."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from openinterpretability.core.schemas import WordStrength
from openinterpretability.harness import specificity


class WordTokenizer:
    """One token for each word or punctuation mark."""

    def __call__(self, text, return_offsets_mapping=False):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]}


@dataclass(frozen=True)
class ShortUnit:
    """Strength 1 on every token, with one strength too few when the text holds ``short``."""

    short: str
    skip_first: bool = False

    def strengths(self, runner, text):
        n = len(runner.tokenizer(text)["offset_mapping"])
        return torch.ones(1, n - 1 if self.short in text else n)


def test_word_strength_is_none_when_strengths_and_tokens_differ_in_length():
    runner = SimpleNamespace(tokenizer=WordTokenizer())
    unit = ShortUnit("odd")
    assert specificity.word_strength(runner, unit, "Here is a word: {}.", "odd") is None
    assert specificity.word_strength(runner, unit, "Here is a word: {}.", "even") == 1.0


def test_measure_leaves_out_words_that_no_carrier_measures():
    runner = SimpleNamespace(tokenizer=WordTokenizer())
    found = specificity.measure(runner, ShortUnit("odd"), ["odd", "even"])
    assert found == [WordStrength(word="even", strength=1.0)]
