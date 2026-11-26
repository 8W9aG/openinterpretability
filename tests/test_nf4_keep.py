"""The helpers of scripts/nf4_keep.py (#5158)."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import nf4_keep as nk  # noqa: E402


def test_each_nf4_value_maps_to_itself():
    levels = torch.tensor(nk.NF4)
    assert torch.equal(nk.nf4_round(levels, block=16), levels)  # its largest value is 1


def test_a_block_is_scaled_by_its_largest_absolute_value():
    block = torch.tensor([0.0, 0.1, -0.5, 2.0])
    rounded = nk.nf4_round(block, block=4)
    assert rounded[3] == 2.0  # the largest value keeps its size
    assert rounded[0] == 0.0  # NF4 has an exact zero
    # -0.5 / 2 = -0.25 is nearest to -0.2844, and 0.1 / 2 = 0.05 nearest to 0.0796.
    assert rounded[2] == pytest.approx(-0.28444138169288635 * 2)
    assert rounded[1] == pytest.approx(0.07958029955625534 * 2)


def test_each_block_has_its_own_scale_and_a_zero_block_stays_zero():
    w = torch.tensor([[1.0, 0.5, 100.0, 50.0], [0.0, 0.0, 0.0, 0.0]])
    rounded = nk.nf4_round(w, block=2)
    assert rounded.shape == w.shape and torch.equal(rounded[1], torch.zeros(4))
    assert rounded[0, 0] == 1.0 and rounded[0, 2] == 100.0
    assert rounded[0, 1] == pytest.approx(0.44070982933044434)  # 0.5 to its nearest level
    assert rounded[0, 3] == pytest.approx(44.070982933044434)  # the same, at its own scale


def test_a_short_last_block_is_padded_and_the_padding_is_cut_off():
    w = torch.arange(1.0, 6.0)  # 5 values in blocks of 4: the last block holds 1 value
    rounded = nk.nf4_round(w, block=4)
    assert rounded.shape == (5,) and rounded[4] == 5.0  # alone in its block: its own scale
    assert rounded[3] == 4.0


def test_the_midpoint_between_two_levels_goes_down():
    levels = torch.tensor(nk.NF4)
    mid = (levels[8] + levels[9]) / 2
    rounded = nk.nf4_round(torch.stack([mid, torch.tensor(1.0)]), block=2)
    assert rounded[0] == levels[8]


class Block(nn.Module):
    """A block with a Qwen2 block's names: attention and MLP weights, a bias and a norm."""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(4, 4, bias=name != "o_proj"))
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 8, bias=False)
        self.mlp.up_proj = nn.Linear(4, 8, bias=False)
        self.mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.input_layernorm = nn.LayerNorm(4)


@pytest.mark.parametrize("condition", nk.CONDITIONS)
def test_each_condition_changes_only_the_linear_weights(condition):
    torch.manual_seed(0)
    block = Block()
    change = nk.change_for(block, condition, seed=1)
    weights = {n: w for n, w in block.named_parameters()}
    assert set(change) == {n for n in weights if n.endswith(".weight") and "norm" not in n}
    for name, tensor in change.items():
        assert tensor.shape == weights[name].shape and tensor.device.type == "cpu"
        error = nk.nf4_round(weights[name]) - weights[name].detach()
        if condition == "nf4":
            assert torch.allclose(tensor, error)
        elif condition == "bf16":
            assert torch.equal(tensor, weights[name].detach().bfloat16().float() - weights[name])
        else:
            assert float(tensor.norm()) == pytest.approx(float(error.norm()), rel=1e-5)
            assert not torch.allclose(tensor, error)


def test_the_random_change_follows_its_seed_and_an_unknown_condition_is_an_error():
    block = Block()
    name = "mlp.up_proj.weight"
    one, again = (nk.change_for(block, "random_nf4", 3)[name] for _ in range(2))
    assert torch.equal(one, again)
    assert not torch.equal(one, nk.change_for(block, "random_nf4", 4)[name])
    with pytest.raises(ValueError, match="unknown condition"):
        nk.change_for(block, "nf8")


def test_a_peak_reads_the_pole_and_can_skip_the_first_token():
    acts = torch.tensor([5.0, -1.0, 2.0])
    assert nk.peak(acts, "positive", False) == 5.0
    assert nk.peak(acts, "positive", True) == 2.0
    assert nk.peak(acts, "negative", True) == 1.0
    assert nk.peak(torch.tensor([3.0]), "positive", True) is None


class _Ident:
    layer, neuron_idx, pole = 0, 1, "positive"


class _Record:
    def __init__(self, prompts, scale):
        results = [type("R", (), {"prompt": p, "kind": "hard_negative"})() for p in prompts]
        self.verdict = type("V", (), {"adversarial_test_results": results})()
        self.submission = type("S", (), {"identifier": _Ident(), "max_activation": scale})()


def test_drift_is_each_probe_peaks_change_as_a_share_of_the_scale():
    record = _Record(["a", "b", "c"], scale=4.0)
    unit = type("U", (), {"skip_first": True})()
    before = {("a", 0, 1): torch.tensor([9.0, 2.0]), ("b", 0, 1): torch.tensor([0.0, 1.0]),
              ("c", 0, 1): torch.tensor([1.0])}  # fmt: skip
    after = {("a", 0, 1): torch.tensor([9.0, 3.0]), ("b", 0, 1): torch.tensor([0.0, 0.0]),
             ("c", 0, 1): torch.tensor([2.0])}  # fmt: skip
    # "c" has one token, which a pole with a first-token floor doesn't read.
    assert nk.drifts(record, unit, before, after) == [0.25, 0.25]


import nf4_keep_stats as stats  # noqa: E402


def test_quantile_interpolates_between_ranks():
    assert stats.quantile([3.0, 1.0, 2.0, 4.0], 0.5) == 2.5
    assert stats.quantile([5.0], 0.99) == 5.0
    assert stats.quantile([0.0, 10.0], 0.99) == pytest.approx(9.9)


def test_drift_rows_pool_the_probes_of_the_poles_that_reproduce():
    rows = [
        {"table": "pole", "model": "m", "condition": "nf4", "seed": None, "base_passes": True,
         "drift": [0.0, 0.02]},
        {"table": "pole", "model": "m", "condition": "nf4", "seed": None, "base_passes": True,
         "drift": [0.06]},
        {"table": "pole", "model": "m", "condition": "nf4", "seed": None, "base_passes": False,
         "drift": [0.9]},
        {"table": "base", "model": "m"},
    ]  # fmt: skip
    (row,) = stats.drift_rows(rows)
    assert row["probes"] == 3 and row["largest"] == 0.06
    assert row["under"] == {"0.01": 0.3333, "0.03": 0.6667, "0.05": 0.6667, "0.07": 1.0}


def test_fisher_matches_known_tables():
    # The lady tasting tea: [[3, 1], [1, 3]] gives p = 34/70.
    assert stats.fisher(3, 1, 1, 3) == pytest.approx(34 / 70)
    assert stats.fisher(4, 0, 0, 4) == pytest.approx(2 / 70)
    assert stats.fisher(0, 0, 2, 2) == 1.0


def test_keep_table_counts_each_group_of_poles():
    normal_rows = [
        {"slug": "a", "p": {"gate_proj": 0.01, "up_proj": 0.5, "down_proj": 0.5}},
        {"slug": "b", "p": {"gate_proj": 0.5, "up_proj": 0.5, "down_proj": 0.01}},
        {"slug": "c", "p": {"gate_proj": 0.5, "up_proj": 0.5, "down_proj": 0.5}},
        {"slug": "d", "p": {"gate_proj": 0.5, "up_proj": 0.5, "down_proj": 0.5}},
    ]
    kept = {"a": False, "b": True, "c": True}  # d did not reproduce
    gate = stats.keep_table(normal_rows, kept, "gate_proj")
    assert (gate["non_normal_fail"], gate["non_normal_keep"]) == (1, 0)
    assert (gate["normal_fail"], gate["normal_keep"]) == (0, 2)
    either = stats.keep_table(normal_rows, kept, "any")
    assert (either["non_normal_fail"], either["non_normal_keep"], either["normal_keep"]) == (
        1,
        1,
        1,
    )
    assert 0 < either["fisher_p"] <= 1


def test_weight_vectors_are_the_poles_rows_and_column():
    mlp = type("M", (), {})()
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = nn.Linear(3, 4), nn.Linear(3, 4), nn.Linear(4, 3)
    layer = type("L", (), {"mlp": mlp})()
    model = type("Q", (), {"model": type("I", (), {"layers": [layer]})()})()
    got = stats.weight_vectors(model, 0, 2)
    assert torch.equal(got["gate_proj"], mlp.gate_proj.weight[2])
    assert torch.equal(got["down_proj"], mlp.down_proj.weight[:, 2])


def test_the_overlap_with_instruction_tuning_counts_the_poles_both_runs_read():
    def pole(slug, condition, keeps, base_passes=True):
        return {"table": "pole", "model": "m", "slug": slug, "condition": condition,
                "keeps": keeps, "base_passes": base_passes}  # fmt: skip

    ours = [pole("a", "nf4", False), pole("b", "nf4", False), pole("c", "nf4", True),
            pole("d", "nf4", True), pole("e", "nf4", False)]  # fmt: skip
    theirs = [pole("a", "instruct", False), pole("b", "instruct", True),
              pole("c", "instruct", False), pole("d", "instruct", True),
              pole("e", "instruct", False, base_passes=False)]  # fmt: skip
    got = stats.instruct_overlap(ours, theirs, "m")
    assert (got["poles"], got["nf4_flips"], got["instruct_flips"], got["both"]) == (4, 2, 2, 1)
    assert got["expected_both"] == 1.0
    assert stats.instruct_overlap([], theirs, "m")["expected_both"] is None
