"""Helpers of the base ↔ instruct comparison (scripts/base_instruct.py, issue #3375)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import base_instruct as bi  # noqa: E402


def test_pole_max_reads_each_pole_as_a_magnitude():
    acts = torch.tensor([-3.0, 1.0, 2.0])
    assert bi.pole_max(acts, "positive") == 2.0
    assert bi.pole_max(acts, "negative") == 3.0


def test_pearson_and_a_constant_vector():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert bi.pearson(a, 2 * a + 1) == pytest.approx(1.0)
    assert bi.pearson(a, -a) == pytest.approx(-1.0)
    assert bi.pearson(a, torch.ones(4)) is None


def test_wilson_interval_brackets_the_share():
    lo, hi = bi.wilson(30, 40)
    assert 0 <= lo < 0.75 < hi <= 1
    assert bi.wilson(0, 0) == [0.0, 1.0]


class _CharTokenizer:
    """One token per character, with a chat template around the user's text."""

    def apply_chat_template(self, turns, tokenize, add_generation_prompt):
        return "<sys>hi</sys><user>" + turns[-1]["content"] + "</user>"

    def __call__(self, text, return_offsets_mapping):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class _Runner:
    tokenizer = _CharTokenizer()


class _Unit:
    """Fires 9 on every template character and 1 + its index within the user's text."""

    def strengths(self, runner, chat):
        start = chat.index("<user>") + len("<user>")
        end = chat.index("</user>")
        return torch.tensor(
            [[float(1 + i - start) if start <= i < end else 9.0 for i in range(len(chat))]]
        )


def test_chat_strength_reads_only_the_probes_tokens():
    # The template's tokens fire at 9, but only the probe's own tokens count.
    assert bi.chat_strength(_Runner(), _Unit(), "abc") == 3.0
