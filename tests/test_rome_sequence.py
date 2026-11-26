"""The helpers of scripts/rome_sequence.py (#5362)."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import rome_sequence as rs  # noqa: E402
import rome_sequence_stats as st  # noqa: E402


class _Tokenizer:
    """Splits on spaces; a word is one token, and an unknown word two."""

    vocab = {" France": 1, " Spain": 2, " Italy": 3, " Germany": 4}

    def __call__(self, text):
        if text in self.vocab:
            return {"input_ids": [self.vocab[text]]}
        return {"input_ids": [hash(w) % 50 + 10 for w in text.split()] + [0]}


def test_facts_never_give_the_true_country_and_use_each_city_once():
    facts = rs.make_facts(_Tokenizer(), n=8, seed=1)
    true = dict(rs.CITIES)
    assert len(facts) == 8 and len({f[0] for f in facts}) == 8
    for city, relation, obj in facts:
        assert relation == rs.RELATION and obj.strip() != true[city]
        assert obj in _Tokenizer.vocab  # a one-token country
    assert facts == rs.make_facts(_Tokenizer(), n=8, seed=1)


def test_too_few_cities_is_an_error():
    with pytest.raises(ValueError, match="fewer than"):
        rs.make_facts(_Tokenizer(), n=10_000)


def test_deltas_sum_the_edits_and_the_null_matches_each_norm():
    edits = [(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0, 0.0])),
             (torch.tensor([0.0, 3.0]), torch.tensor([1.0, 0.0, 0.0]))]  # fmt: skip
    rome = rs.deltas(edits, "rome")
    assert torch.equal(rome[0], torch.outer(*edits[0]))
    assert torch.equal(rome[1], torch.outer(*edits[0]) + torch.outer(*edits[1]))
    null = rs.deltas(edits, "random")
    first, second = null[0], null[1] - null[0]
    assert float(first.norm()) == pytest.approx(2.0, rel=1e-5)
    assert float(second.norm()) == pytest.approx(3.0, rel=1e-5)
    assert not torch.allclose(first, rome[0])


def test_distance_is_relative_to_the_base_weight():
    weight = torch.ones(2, 2)  # norm 2
    assert rs.distance(torch.zeros(2, 2), weight) == 0.0
    assert rs.distance(torch.ones(2, 2), weight) == pytest.approx(1.0)


class _Model(torch.nn.Module):
    """Always predicts token 2, with a fixed loss."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids, labels=None):
        logits = torch.zeros(ids.shape[0], ids.shape[1], 5)
        logits[..., 2] = 1.0
        out = type("Out", (), {})()
        out.logits, out.loss = logits, torch.tensor(1.5)
        return out


def test_top_share_counts_the_facts_whose_new_object_wins():
    facts = [("Paris", rs.RELATION, " Spain"), ("Rome", rs.RELATION, " Italy")]
    assert rs.top_share(_Model(), _Tokenizer(), facts) == 0.5
    assert rs.top_share(_Model(), _Tokenizer(), []) == 0.0


def test_sample_loss_is_the_mean_over_batches():
    assert rs.sample_loss(_Model(), [torch.zeros(1, 4, dtype=torch.long)] * 3) == 1.5


def _pole(kind, n, slug, keeps, layer=3):
    return {"table": "pole", "condition": kind, "edits": n, "slug": slug, "layer": layer,
            "base_passes": True, "keeps": keeps}  # fmt: skip


def _check(kind, n, keeps, d):
    return {"table": "checkpoint", "condition": kind, "edits": n, "keeps_above": keeps,
            "poles_above": 4, "distance": d}  # fmt: skip


def test_above_keeps_only_poles_above_the_layer_that_reproduce():
    rows = [_pole("rome", 0, "a", True), _pole("rome", 0, "b", True, layer=2),
            {**_pole("rome", 0, "c", False), "base_passes": False}]  # fmt: skip
    assert st.above(rows, 2) == {("rome", 0): {"a": True}}


def test_paired_counts_the_discordant_poles():
    keeps = {("rome", 5): {"a": True, "b": True, "c": False},
             ("random", 5): {"a": False, "b": True, "c": False}}  # fmt: skip
    (row,) = st.paired(keeps, [5])
    assert (row["rome_only"], row["random_only"], row["p"]) == (1, 0, 1.0)


def test_rate_is_poles_lost_per_unit_distance():
    rows = st.rate([_check("rome", 0, 4, 0.0), _check("rome", 1, 2, 0.5),
                    _check("rome", 2, 2, 0.5)])  # fmt: skip
    assert rows[0]["lost"] == 2 and rows[0]["per_distance"] == 4.0
    assert rows[1]["per_distance"] is None  # no step in distance


def test_hazard_skips_checkpoints_where_nothing_keeps():
    rows = [_check("x", 0, 4, 0.0), _check("x", 1, 2, 1.0), _check("x", 2, 0, 3.0)]
    got = st.hazard(rows)
    assert got["points"] == 1 and got["h"] == pytest.approx(math.log(2), abs=1e-4)


def test_hazard_interval_brackets_the_rate():
    slugs = [f"p{i}" for i in range(20)]
    keeps = {("rome", 0): dict.fromkeys(slugs, True),
             ("rome", 1): {s: i < 10 for i, s in enumerate(slugs)},
             ("rome", 2): {s: i < 5 for i, s in enumerate(slugs)}}  # fmt: skip
    rows = [{**_check("rome", n, k, d), "poles_above": 20}
            for n, k, d in ((0, 20, 0.0), (1, 10, 1.0), (2, 5, 2.0))]  # fmt: skip
    low, high = st.hazard_interval(keeps, rows, draws=200)
    assert low <= st.hazard(rows)["h"] <= high


def test_layers_and_returns():
    rows = [_pole("rome", 0, "a", True), _pole("rome", 1, "a", False),
            _pole("rome", 2, "a", True), _pole("random", 0, "a", True),
            _pole("random", 1, "a", False), _pole("random", 2, "a", False)]  # fmt: skip
    by_layer = st.layers(rows, 2)
    assert {"table": "layers", "condition": "rome", "edits": 1, "layer": 3, "keeps": 0,
            "poles": 1} in by_layer  # fmt: skip
    back = {r["condition"]: r["poles"] for r in st.returns(st.above(rows, 2), [0, 1, 2])}
    assert back == {"rome": 1, "random": 0}


def test_curve_carries_the_interval():
    row = {**_check("rome", 3, 2, 1.0), "keeps_below": 9, "loss": 3.5, "retention": 0.5,
           "latest": 1.0}  # fmt: skip
    (got,) = st.curve([row])
    assert got["ci95"][0] < 0.5 < got["ci95"][1] and got["keeps_below"] == 9
