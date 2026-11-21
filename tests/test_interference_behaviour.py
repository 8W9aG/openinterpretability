"""Behaviour measures of a change: gate masks, explained prompts, answer probabilities."""

from datetime import UTC, datetime

import torch

from openinterpretability.interference.behaviour import (
    _open_mask,
    answer_probabilities,
    explained_prompts,
)
from openinterpretability.interference.floors import Plain
from tests.test_explain_reuse import INPUTS
from tests.test_explanation_records import _record, _write
from tests.test_interference import _runner


def test_a_scalar_gate_opens_or_shuts_every_token():
    shape = torch.Size([2, 3])
    assert _open_mask(torch.tensor(0.7), shape).tolist() == [[True] * 3] * 2
    assert _open_mask(torch.tensor(0.2), shape).tolist() == [[False] * 3] * 2


def test_explained_prompts_are_the_models_prompts_once_each(tmp_path):
    first = _record(datetime(2026, 11, 1, tzinfo=UTC)).model_copy(update={"inputs": INPUTS})
    again = _record(datetime(2026, 11, 2, tzinfo=UTC))  # the same prompt, explained again
    other = _record(datetime(2026, 11, 3, tzinfo=UTC)).model_copy(
        update={"prompt": "The capital of France is", "answer": " Paris"}
    )  # no inputs, so no token id
    elsewhere = _record(datetime(2026, 11, 4, tzinfo=UTC)).model_copy(
        update={"model_name": "other/model"}
    )
    for record in (first, again, other, elsewhere):
        _write(tmp_path, record)
    found = explained_prompts(tmp_path, "openai-community/gpt2")
    assert sorted(found, key=lambda p: p["prompt"]) == [
        {"prompt": "John gave a drink to", "position": -1, "answer": " Mary", "token_id": 1},
        {"prompt": "The capital of France is", "position": -1, "answer": " Paris",
         "token_id": None},
    ]  # fmt: skip
    assert explained_prompts(tmp_path, "no/model") == []


def test_answer_probabilities_read_the_answer_tokens_probability(tmp_path):
    runner = _runner()
    prompts = [
        {"prompt": "abc", "position": -1, "token_id": 5},
        {"prompt": "abc", "position": -1, "token_id": None},
    ]
    ids = torch.tensor([runner.tokenizer.encode("abc")])
    with torch.no_grad():
        logits = runner.model(input_ids=ids).logits[0, -1].float()
    expected = round(float(torch.softmax(logits, -1)[5]), 6)
    assert answer_probabilities(Plain(runner, "base"), prompts) == [expected, None]
    assert 0 < expected < 1
