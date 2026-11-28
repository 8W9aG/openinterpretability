"""The chat-template readings pass each argument on and read the user's own tokens (#6491)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from openinterpretability.agents import review_measurements, reviewer
from openinterpretability.agents.reviewer import AdversarialProbe
from openinterpretability.core.schemas import AdversarialTestResult, NeuronIdentifier, ReviewPolicy
from openinterpretability.interference import keep, template
from tests.conftest import FakeTokenizer


class StrictChat(FakeTokenizer):
    """A chat template that quotes the text in its system turn, and checks each call's options.

    The text appears twice, so only ``rindex`` finds the user's own turn.
    """

    def apply_chat_template(self, turns, *, tokenize, add_generation_prompt):
        (turn,) = turns
        assert turn == {"role": "user", "content": turn.get("content")}
        assert tokenize is False and add_generation_prompt is False
        return f"{turn['content']}<s>{turn['content']}</u>"

    def __call__(self, texts, return_offsets_mapping=False, **options):
        if isinstance(texts, str):
            assert return_offsets_mapping is True
        return super().__call__(texts, return_offsets_mapping=return_offsets_mapping, **options)


class Unit:
    def __init__(self, strengths):
        self.values = strengths

    def strengths(self, _runner, _chat):
        return [torch.tensor(self.values)]


def test_chat_strength_reads_only_the_user_turns_own_tokens():
    runner = SimpleNamespace(tokenizer=StrictChat())
    # "ab<s>ab</u>": the user's "ab" is at 5 and 6. Strengths stop before 6.
    unit = Unit([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    assert keep.chat_strength(runner, unit, "ab") == 5.0
    assert keep.chat_strength(runner, unit, "") is None  # no token overlaps


def test_chat_probes_pass_the_template_and_keep_each_probes_fields(monkeypatch):
    seen = []

    def strength(runner, unit, text, template):
        seen.append(template)
        return {"on": 0.12345678, "off": None}[text]

    monkeypatch.setattr(keep, "chat_strength", strength)
    probes = [AdversarialProbe(text="on", expected_active=True, rationale="why"),
              AdversarialProbe(text="off", expected_active=False, rationale="no")]  # fmt: skip
    results, _, _ = keep.chat_probes("runner", "unit", probes, 1.0, 0.1, template="T")
    assert seen == ["T", "T"]
    on, off = results
    assert (on.expected_active, on.activation, on.measured, on.detail) == (True, 0.123457, True,
                                                                           "why")  # fmt: skip
    assert (off.expected_active, off.activation, off.measured) == (False, None, False)


def _record():
    probe = AdversarialTestResult(kind="hard_negative", prompt="x", passed=True,
                                  expected_active=False, detail="old")  # fmt: skip
    verdict = SimpleNamespace(policy=ReviewPolicy(), adversarial_test_results=[probe])
    return SimpleNamespace(verdict=verdict, submission="sub")


@pytest.mark.parametrize("chat", [False, True])
def test_readings_pass_the_runner_policy_scale_and_template_on(monkeypatch, chat):
    calls = {}

    def settle(submission, runner, policy):
        calls["settle"] = (submission, runner, policy)
        return None, "unit", 2.0

    def probes(*args):
        calls["probes"] = args
        return [], None, None

    def metrics(results, policy, independent):
        calls["independent"] = independent
        return SimpleNamespace(specificity=1.0, sensitivity=0.5, balanced_accuracy=0.75,
                               gates={g: g != keep.GATES[0] for g in keep.GATES})  # fmt: skip

    monkeypatch.setattr(review_measurements, "settle_first_token_floor", settle)
    monkeypatch.setattr(review_measurements, "measure_probes", probes)
    monkeypatch.setattr(keep, "chat_probes", probes)
    monkeypatch.setattr(reviewer, "review_metrics", metrics)
    record = _record()
    extra = {"chat": True, "template": "T"} if chat else {}
    out = keep.readings("runner", record, **extra)
    policy = record.verdict.policy
    assert calls["settle"] == ("sub", "runner", policy) and calls["independent"] is True
    runner, unit, (probe,), scale, threshold, *rest = calls["probes"]
    assert (runner, unit, scale, threshold) == ("runner", "unit", 2.0,
                                                policy.activation_threshold * 2.0)  # fmt: skip
    assert rest == (["T"] if chat else []) and probe.rationale == ""
    assert out["passes"] is False and out["failed"] == [keep.GATES[0]]


def test_pearson_is_rounded_to_four_places():
    assert keep.pearson(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0])) == 0.982


def test_sample_stats_read_each_poles_own_column(monkeypatch, capsys):
    columns = {"base": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
               "instruct": torch.tensor([[3.0, 1.0], [1.0, 0.0]])}  # fmt: skip
    monkeypatch.setattr(keep, "layer_columns", lambda runner, b, layer, neurons: columns[runner])
    idents = [NeuronIdentifier(model_name="m", layer=1, neuron_idx=n, pole="positive")
              for n in (2, 5)]  # fmt: skip
    records = [SimpleNamespace(submission=SimpleNamespace(identifier=i)) for i in idents]
    out = keep.sample_stats("base", "instruct", [], records)
    assert out == {idents[0].slug: {"ratio": 1.5, "pearson": -1.0},
                   idents[1].slug: {"ratio": None, "pearson": None}}  # fmt: skip
    assert "layer 1: 2 poles" in capsys.readouterr().err


def test_chat_sample_stops_once_it_holds_enough_document_tokens(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in ("ab", "cd", "ef")))
    sample = template.chat_sample(StrictChat(), StrictChat(), corpus, tokens=4)
    assert len(sample) == 2  # 2 + 2 tokens reach 4 exactly
    assert sample[0]["text"].tolist() == [False] * 5 + [True] * 2 + [False] * 4
