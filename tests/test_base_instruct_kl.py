"""``scripts/base_instruct_kl.py``: the output KL at each pole's texts, against its keep (#5141)."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "base_instruct_kl", Path(__file__).parents[1] / "scripts" / "base_instruct_kl.py"
)
kl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kl)


class Tokens:
    """A tokenizer with one token per character, and a chat template that wraps the turn."""

    pad_token_id = 0

    def __call__(self, text, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids):
        return "".join(map(chr, ids))

    def apply_chat_template(self, turns, tokenize, add_generation_prompt):
        assert not tokenize and not add_generation_prompt
        return f"<u>{turns[0]['content']}</u>"


def test_the_mean_kl_counts_only_the_kept_positions():
    p = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 3.0]])
    q = torch.zeros(3, 2)
    one = (torch.softmax(p[0], 0) * (torch.log_softmax(p[0], 0) - math.log(0.5))).sum()
    assert math.isclose(kl.mean_kl(p, q, [True, False, False]), float(one), rel_tol=1e-6)
    assert kl.mean_kl(p, p) == 0.0
    assert math.isnan(kl.mean_kl(p, q, [False, False, False]))  # no position: no mean


def test_a_long_text_is_cut_to_its_first_tokens(monkeypatch):
    monkeypatch.setattr(kl, "MAX_TOKENS", 4)
    assert kl.clipped(Tokens(), "abcdefg") == "abcd" and kl.clipped(Tokens(), "abcd") == "abcd"


def test_plain_texts_are_right_padded_and_count_every_real_position(monkeypatch):
    ids, flags = kl.encoded(Tokens(), ["ab", "c"])
    assert ids.tolist() == [[97, 98], [99, 0]]
    assert flags == [[True, True], [True, False]]  # the pad never counts


def test_a_chat_turn_counts_only_the_users_text():
    ids, flags = kl.encoded(Tokens(), ["hi"], template=Tokens())
    assert ids.shape == (1, len("<u>hi</u>"))
    assert flags == [[False, False, False, True, True, False, False, False, False]]


def test_texts_kl_reads_both_models_per_batch_and_keeps_text_order(monkeypatch):
    calls = []

    class Runner:
        tokenizer = Tokens()

        def __init__(self, bias):
            self.bias = bias

        def run_with_interventions(self, ids, interventions):
            assert interventions == {}
            calls.append(ids.shape[0])
            logits = torch.zeros(*ids.shape, 2)
            logits[..., 0] = self.bias * (ids > 98).float()  # differ only on letters after 'b'
            return logits

    monkeypatch.setattr(kl, "BATCH_POSITIONS", 40)  # each text is 1 + 32: one per batch
    base, instruct = Runner(0.0), Runner(2.0)
    found = kl.texts_kl(base, instruct, ["a", "c"])
    assert found[0] == 0.0 and found[1] > 0.3 and calls == [1, 1, 1, 1]
    assert kl.texts_kl(base, instruct, []) == []


def test_batches_fill_up_to_the_budget():
    assert kl.batches([3, 3, 3, 5], budget=9) == [[0, 1, 2], [3]]
    assert kl.batches([2, 2], budget=4) == [[0, 1]]
    assert kl.batches([10, 1], budget=8) == [[0], [1]]


def test_a_poles_texts_are_its_probes_and_its_top_contexts():
    results = [NS(kind="hard_negative", prompt="n"), NS(kind="positive_holdout", prompt="p"),
               NS(kind="counterfactual", prompt="c")]  # fmt: skip
    record = NS(verdict=NS(adversarial_test_results=results),
                submission=NS(top_examples=[NS(text="t1"), NS(text="t2")]))  # fmt: skip
    assert kl.pole_texts(record) == {"probes": ["n", "p"], "top": ["t1", "t2"]}


def test_a_pole_row_has_its_four_measures(monkeypatch):
    seen = []

    def fake(base, instruct, texts, template=None):
        seen.append((tuple(texts), template is None))
        return [0.25, float("nan")] if texts == ["n", "p"] else []

    monkeypatch.setattr(kl, "texts_kl", fake)
    monkeypatch.setattr(kl, "pole_texts", lambda record: {"probes": ["n", "p"], "top": []})
    record = NS(submission=NS(identifier=NS(slug="m/L1/N2-positive", layer=1)))
    row = kl.pole_row(None, NS(tokenizer="instruct"), record)
    assert row == {"table": "pole", "slug": "m/L1/N2-positive", "layer": 1,
                   "kl_probes_plain": 0.25, "kl_probes_chat": 0.25,
                   "kl_top_plain": None, "kl_top_chat": None}  # fmt: skip
    assert seen[:2] == [(("n", "p"), True), (("n", "p"), False)]


def test_keep_labels_skip_poles_that_did_not_reproduce(tmp_path):
    rows = [{"table": "summary"},
            {"table": "pole", "slug": "a", "keeps": True, "base": {"passes": True}},
            {"table": "pole", "slug": "b", "keeps": False, "base": {"passes": True}},
            {"table": "pole", "slug": "c", "keeps": False, "base": {"passes": False}}]  # fmt: skip
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")
    assert kl.keep_labels(path) == {"a": True, "b": False}


def test_each_measure_and_mode_gets_an_auroc_and_a_shuffle_p():
    poles = [{"slug": s, "layer": 0, "kl_probes_plain": v, "kl_probes_chat": v,
              "kl_top_plain": None if s == "d" else v, "kl_top_chat": v}
             for s, v in (("a", 0.1), ("b", 0.2), ("c", 0.9), ("d", 0.8))]  # fmt: skip
    keeps = {"a": True, "b": True, "c": False, "d": False}
    rows = kl.auroc_rows("m", poles, {"plain": keeps, "chat": keeps})
    assert [(r["measure"], r["mode"]) for r in rows] == [
        ("probes", "plain"),
        ("probes", "chat"),
        ("top", "plain"),
        ("top", "chat"),
    ]
    first = rows[0]
    assert (first["base"], first["poles"], first["fails"], first["auroc"]) == ("m", 4, 2, 1.0)
    assert rows[2]["poles"] == 3  # a pole with no value is left out
    assert 0 < first["p_within_layer_shuffle"] <= 1
