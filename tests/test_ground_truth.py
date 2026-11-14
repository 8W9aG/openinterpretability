"""The label rules and gate behind docs/analyses/ground-truth-calibration.md."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from ground_truth import gate_scores, label, word_ids  # noqa: E402


class Pieces:
    """A tokenizer whose ids are indices into a fixed list of token strings."""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)

    def encode(self, text):
        return [self.pieces.index(text)] if text in self.pieces else [0, 0]


def ids_of(tok, *pieces):
    return [tok.pieces.index(p) for p in pieces]


def test_word_ids_take_single_tokens_in_each_form():
    tok = Pieces(["?", "May", " May", " may", " June"])
    assert word_ids(tok, "May") == {1, 2, 3}
    assert word_ids(tok, "July") == set()


def test_token_set_rules():
    tok = Pieces(["?", " In", " May", " the", " cat", " sat", " tea"])
    ids = ids_of(tok, " In", " May", " the", " cat", " sat", " tea")
    assert label("months", tok, ids) == [False, True, False, False, False, False]
    assert label("animals", tok, ids) == [False, False, False, True, False, False]
    assert label("weekdays_or_drinks", tok, ids) == [False] * 5 + [True]


def test_after_title_takes_the_name_and_skips_the_full_stop():
    tok = Pieces(["?", " Mr", ".", " Smith", " and", " Dr", " Jones"])
    ids = ids_of(tok, " Mr", ".", " Smith", " and", " Dr", " Jones")
    assert label("after_title", tok, ids) == [False, False, True, False, False, True]


def test_in_quotes_marks_only_the_tokens_between_the_marks():
    tok = Pieces(["?", " He", ' "', "Hi", " there", '"', " now", " “", "Go", "”"])
    ids = ids_of(tok, " He", ' "', "Hi", " there", '"', " now", " “", "Go", "”")
    assert label("in_quotes", tok, ids) == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_line_start_is_the_token_after_a_newline():
    tok = Pieces(["?", " a", "\n", "b", "\n\n", " ", "c"])
    ids = ids_of(tok, " a", "\n", "b", "\n\n", " ", "c")
    assert label("line_start", tok, ids) == [False, False, True, False, False, False]


def test_gate_scores_on_the_test_tokens_at_the_line():
    act = torch.tensor([4.0, 0.0, 3.0, 0.5, 4.0, 0.0])
    y = torch.tensor([True, False, True, True, False, False])
    test = torch.tensor([True, True, True, True, False, False])
    scores = gate_scores(act, y, test, scale=4.0)  # line 1.0
    assert scores["precision"] == 1.0 and abs(scores["recall"] - 2 / 3) < 1e-9
    assert scores["active_share"] == 0.5 and scores["positives"] == 3
