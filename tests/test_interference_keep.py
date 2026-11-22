"""The keep test of a VERIFIED pole on a changed model (interference/keep.py, #5230)."""

from __future__ import annotations

import json

import pytest
import torch

from openinterpretability.agents.reviewer import AdversarialProbe
from openinterpretability.builder.wiki import archive_record
from openinterpretability.core.schemas import AdversarialTestResult, NeuronIdentifier
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.interference import keep
from tests.conftest import FakeTokenizer, ToyModel


def _pole(record, neuron, pole="positive", status="VERIFIED", model="Qwen/Qwen2.5-0.5B"):
    ident = NeuronIdentifier(model_name=model, layer=1, neuron_idx=neuron, pole=pole)
    sub = record.submission.model_copy(
        update={"identifier": ident, "max_activation": 1.0, "first_token_floor": False}
    )
    probes = [AdversarialTestResult(kind=kind, prompt=text, passed=True, expected_active=active)
              for kind, text, active in [("positive_holdout", "abc abc abc", True),
                                         ("hard_negative", "xyz xyz", False)]]  # fmt: skip
    verdict = record.verdict.model_copy(
        update={"status": status, "adversarial_test_results": probes}
    )
    return record.model_copy(update={"submission": sub, "verdict": verdict})


def _runner(seed=0):
    torch.manual_seed(seed)
    return NeuronHookRunner(ToyModel().eval(), ChatTokenizer())


class ChatTokenizer(FakeTokenizer):
    """The toy tokenizer, with a chat template around the user's text."""

    def apply_chat_template(self, turns, tokenize, add_generation_prompt):
        return "<sys>" + turns[-1]["content"] + "</u>"


def test_verified_records_are_the_models_verified_neuron_poles_only(record, tmp_path):
    wiki = tmp_path / "wiki"
    archive_record(_pole(record, 3), wiki)
    archive_record(_pole(record, 4, status="REJECTED"), wiki)
    archive_record(_pole(record, 5, model="EleutherAI/pythia-70m"), wiki)
    found = keep.verified_records(wiki, "Qwen/Qwen2.5-0.5B")
    assert [r.submission.identifier.neuron_idx for r in found] == [3]


def test_sample_windows_cuts_the_corpus_into_batches_of_whole_windows(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    lines = [json.dumps({"text": "a" * 400}) for _ in range(10)]
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    batches = keep.sample_windows(FakeTokenizer(), tokens=keep.WINDOW * 5 + 10, corpus=corpus)
    assert [b.shape for b in batches] == [(4, keep.WINDOW), (1, keep.WINDOW)]


def test_a_corpus_shorter_than_the_sample_gives_its_whole_windows(tmp_path):
    # #5909: 519 tokens and a sample of 2,048 used to keep all 519 and fail the reshape.
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a" * (keep.WINDOW + 7)}) + "\n", encoding="utf-8")
    batches = keep.sample_windows(FakeTokenizer(), tokens=keep.WINDOW * 4, corpus=corpus)
    assert [b.shape for b in batches] == [(1, keep.WINDOW)]


def test_a_corpus_shorter_than_one_window_is_refused(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a" * 10}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"holds 10 tokens, less than one window of {keep.WINDOW}"):
        keep.sample_windows(FakeTokenizer(), tokens=keep.WINDOW, corpus=corpus)


def test_sample_stats_give_ratio_and_correlation_per_pole(record):
    records = [_pole(record, 3), _pole(record, 7, pole="negative")]
    torch.manual_seed(2)
    batches = [torch.randint(1, 16, (2, 12))]
    base = _runner(0)
    same = keep.sample_stats(base, _runner(0), batches, records)
    assert set(same) == {r.submission.identifier.slug for r in records}
    for stats in same.values():
        assert stats["ratio"] == pytest.approx(1.0) and stats["pearson"] == pytest.approx(1.0)
    other = keep.sample_stats(base, _runner(1), batches, records)
    assert all(s["ratio"] != pytest.approx(1.0) for s in other.values())
    assert all(-1.0 <= s["pearson"] <= 1.0 for s in other.values())


class _Unit:
    """Fires 1 + the index of each character of the user's text, and 0 on the template."""

    def strengths(self, runner, chat):
        start, end = chat.index("<sys>") + 5, chat.index("</u>")
        return torch.tensor([[float(1 + i - start) if start <= i < end else 0.0
                              for i in range(len(chat))]])  # fmt: skip


def test_chat_probes_read_each_probe_inside_the_template():
    probes = [AdversarialProbe(text="abcd", expected_active=True, rationale="long"),
              AdversarialProbe(text="ab", expected_active=False, rationale="short"),
              AdversarialProbe(text="abc", expected_active=False, rationale="near")]  # fmt: skip
    results, low, high = keep.chat_probes(_runner(), _Unit(), probes, 1.0, 3.0)
    assert (low, high) == (None, None)
    assert [r.kind for r in results] == ["positive_holdout", "hard_negative", "hard_negative"]
    assert [r.activation for r in results] == [4.0, 2.0, 3.0]
    assert [r.passed for r in results] == [True, True, False]
    assert [r.detail for r in results] == ["long", "short", "near"]
    assert all(r.measured for r in results)


def test_chat_probes_count_nothing_active_at_zero_scale():
    probe = AdversarialProbe(text="abcd", expected_active=True, rationale="")
    (result,), _, _ = keep.chat_probes(_runner(), _Unit(), [probe], 0.0, 0.0)
    assert result.activation == 4.0 and not result.passed


def test_readings_in_the_chat_template_report_both_gates(record):
    out = keep.readings(_runner(), _pole(record, 3), chat=True)
    assert set(out) == {"specificity", "sensitivity", "balanced_accuracy", "passes", "failed"}
    assert out["passes"] == (out["failed"] == [])
    assert set(out["failed"]) <= set(keep.GATES)
