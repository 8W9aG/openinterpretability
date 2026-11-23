"""A corpus sample inside a chat template, and the template-or-tuning script (#5126)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.interference import keep, template
from tests.conftest import FakeTokenizer, ToyModel


class Render(FakeTokenizer):
    """A template with a system turn around the user's text."""

    def __init__(self, system: str = "<s>"):
        self.system = system

    def apply_chat_template(self, turns, tokenize, add_generation_prompt):
        return self.system + turns[-1]["content"] + "</u>"


def _runner(seed=0):
    torch.manual_seed(seed)
    return NeuronHookRunner(ToyModel().eval(), FakeTokenizer())


def test_chat_sample_marks_the_documents_own_tokens(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in ("abc", "  ", "defg")))
    sample = template.chat_sample(Render(), FakeTokenizer(), corpus, tokens=5)
    assert len(sample) == 2  # the blank document is skipped, and 3 + 4 tokens reach 5
    first = sample[0]
    assert first["ids"].shape == (1, len("<s>abc</u>"))
    assert first["text"].tolist() == [False] * 3 + [True] * 3 + [False] * 4


def test_chat_sample_refuses_a_corpus_with_no_text(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": " "}) + "\n")
    with pytest.raises(ValueError, match="no document tokens"):
        template.chat_sample(Render(), FakeTokenizer(), corpus)


def test_chat_columns_split_text_and_template_tokens(tmp_path):
    runner = _runner()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "abcd"}) + "\n")
    sample = template.chat_sample(Render(), FakeTokenizer(), corpus, tokens=4)
    cols = template.chat_columns(runner, sample, {1: [2, 0]})
    assert cols[1]["text"].shape == (4, 2) and cols[1]["template"].shape == (7, 2)
    full = runner.layer_activations({"input_ids": sample[0]["ids"]}, [1])[1][0]
    assert torch.allclose(cols[1]["text"][:, 1], full[3:7, 0])


def test_the_chat_reading_can_use_another_models_template():
    runner = _runner()
    unit = SimpleNamespace(strengths=lambda r, text: [torch.arange(len(text), dtype=torch.float)])
    own = keep.chat_strength(runner, unit, "ab", Render("<s>"))
    other = keep.chat_strength(runner, unit, "ab", Render("<longer>"))
    assert own == 4.0 and other == 9.0  # the text's last token, after each system prompt


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "template_tuning.py"
    spec = importlib.util.spec_from_file_location("template_tuning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(passes, spec=1.0, sens=1.0):
    return {"passes": passes, "specificity": spec, "sensitivity": sens}


def test_compare_counts_each_change_with_paired_tests():
    script = _script()
    rows = [
        {"base_plain": _read(True), "base_chat": _read(True, sens=0.8),
         "instruct_plain": _read(True), "instruct_chat": _read(False, sens=0.5)},
        {"base_plain": _read(True), "base_chat": _read(False, sens=0.5),
         "instruct_plain": _read(False, spec=0.7), "instruct_chat": _read(False, sens=0.5)},
        {"base_plain": _read(False), "base_chat": _read(True),
         "instruct_plain": _read(True), "instruct_chat": _read(True)},  # left out: no base keep
    ]  # fmt: skip
    changes = {c["change"]: c for c in script.compare(rows)}
    assert changes["template"]["lost"] == 1 and changes["template"]["poles"] == 2
    assert changes["tuning_plain"]["specificity"] == {"mean_change": -0.15, "down": 1, "up": 0,
                                                      "p_sign": 1.0}  # fmt: skip
    assert changes["tuning_chat"]["lost"] == 1 and changes["tuning_chat"]["gained"] == 0
    head = script.summary(rows, "M")
    assert head["reproduced"] == 2 and head["keeps"]["instruct_chat"] == 0


def test_binomial_p_is_two_sided_and_capped():
    script = _script()
    assert script.binomial_p(0, 0) == 1.0
    assert script.binomial_p(5, 10) == 1.0
    assert script.binomial_p(0, 5) == pytest.approx(2 / 32)


def test_chat_stats_reads_each_pole_on_text_and_template(tmp_path):
    script = _script()
    base, instruct = _runner(0), _runner(1)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "abcdefgh"}) + "\n")
    sample = template.chat_sample(Render(), FakeTokenizer(), corpus, tokens=8)
    ident = lambda n, pole: SimpleNamespace(layer=1, neuron_idx=n, pole=pole,  # noqa: E731
                                            slug=f"M/L1/N{n}-{pole}")  # fmt: skip
    records = [SimpleNamespace(submission=SimpleNamespace(identifier=ident(2, p)))
               for p in ("positive", "negative")]  # fmt: skip
    stats = script.chat_stats(base, instruct, sample, records)
    pos, neg = stats["M/L1/N2-positive"], stats["M/L1/N2-negative"]
    assert pos["pearson_chat"] == neg["pearson_chat"]  # one neuron, one correlation
    assert pos["template_base"] >= 0 and neg["template_base"] >= 0
