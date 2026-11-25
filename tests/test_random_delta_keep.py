"""The helpers of scripts/random_delta_keep.py (#5154)."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import random_delta_keep as rdk  # noqa: E402


class Block(nn.Module):
    """A block with a Qwen2 MLP's names, and one other weight."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(3, 4, bias=False)
        self.mlp.up_proj = nn.Linear(3, 4, bias=False)
        self.mlp.down_proj = nn.Linear(4, 3, bias=False)
        self.attn = nn.Linear(3, 3)


def _pair():
    torch.manual_seed(0)
    base, other = Block(), Block()
    return base, other


def test_deltas_are_the_other_weights_minus_the_base():
    base, other = _pair()
    delta = rdk.deltas(base, other)
    assert set(delta) == {n for n, _ in base.named_parameters()}
    for name, weight in base.named_parameters():
        assert torch.equal(delta[name], dict(other.named_parameters())[name] - weight)


def test_mlp_names_are_the_three_matrices():
    assert rdk.is_mlp("model.layers.3.mlp.down_proj.weight")
    assert not rdk.is_mlp("model.layers.3.self_attn.q_proj.weight")
    assert not rdk.is_mlp("model.layers.3.mlp.down_proj.bias")


@pytest.mark.parametrize(
    ("condition", "names"),
    [
        ("instruct", {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                      "attn.weight", "attn.bias"}),
        ("instruct_mlp", {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}),
        ("random_mlp", {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}),
        ("random_all", {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                        "attn.weight", "attn.bias"}),
    ],
)  # fmt: skip
def test_each_condition_changes_its_own_weights(condition, names):
    delta = rdk.deltas(*_pair())
    change = rdk.change_for(condition, delta, seed=1)
    assert set(change) == names
    for name, tensor in change.items():
        assert tensor.shape == delta[name].shape
        assert float(tensor.norm()) == pytest.approx(float(delta[name].norm()), rel=1e-5)
        if condition.startswith("instruct"):
            assert torch.equal(tensor, delta[name])
        else:
            assert not torch.equal(tensor, delta[name])


def test_random_changes_follow_their_seed():
    delta = rdk.deltas(*_pair())
    one, again = rdk.change_for("random_mlp", delta, 3), rdk.change_for("random_mlp", delta, 3)
    other = rdk.change_for("random_mlp", delta, 4)
    name = "mlp.up_proj.weight"
    assert torch.equal(one[name], again[name]) and not torch.equal(one[name], other[name])


def test_an_unknown_condition_is_an_error():
    with pytest.raises(ValueError, match="unknown condition"):
        rdk.change_for("instruct_attention", {}, 0)


def test_the_weights_come_back_exactly_after_a_change():
    base, other = _pair()
    before = {n: w.detach().clone() for n, w in base.named_parameters()}
    change = rdk.change_for("instruct", rdk.deltas(base, other))
    with rdk.changed(base, change) as model:
        for name, weight in model.named_parameters():
            assert torch.allclose(weight, dict(other.named_parameters())[name], atol=1e-6)
    for name, weight in base.named_parameters():
        assert torch.equal(weight, before[name])


def test_the_weights_come_back_after_an_error():
    base, other = _pair()
    before = {n: w.detach().clone() for n, w in base.named_parameters()}
    with (
        pytest.raises(RuntimeError),
        rdk.changed(base, rdk.change_for("random_all", rdk.deltas(base, other))),
    ):
        raise RuntimeError("a failed condition")
    for name, weight in base.named_parameters():
        assert torch.equal(weight, before[name])


def test_summary_counts_only_the_poles_that_reproduce():
    rows = [
        {"slug": "a", "base_passes": True, "keeps": True},
        {"slug": "b", "base_passes": True, "keeps": False},
        {"slug": "c", "base_passes": False, "keeps": False},
    ]
    out = rdk.summary(rows, condition="instruct")
    assert out["poles"] == 3 and out["reproduced_on_base"] == 2 and out["keeps"] == 1
    assert out["keep_share"] == 0.5 and out["flipped"] == ["b"] and out["condition"] == "instruct"
    assert rdk.summary([], condition="x")["keep_share"] is None


class _Ident:
    def __init__(self, layer, neuron, pole):
        self.layer, self.neuron_idx, self.pole = layer, neuron, pole
        self.slug = f"L{layer}-N{neuron}-{pole}"


class _Record:
    def __init__(self, layer, neuron, pole):
        self.submission = type("S", (), {"identifier": _Ident(layer, neuron, pole)})()


def test_run_condition_reads_each_pole_on_the_changed_model(monkeypatch):
    base_model, other = _pair()
    runner = type("R", (), {"model": base_model})()
    records = [_Record(0, 1, "positive"), _Record(0, 2, "negative"), _Record(1, 0, "positive")]
    columns = {0: torch.tensor([[1.0, -2.0], [3.0, 0.5]]), 1: torch.tensor([[0.0], [-1.0]])}
    changed_by = {"on": False}

    def fake_columns(_runner, _batches, layer, neurons):
        # The changed model doubles every activation, so each ratio is 2 where a pole has a peak.
        return columns[layer][:, : len(neurons)] * (2.0 if changed_by["on"] else 1.0)

    monkeypatch.setattr(rdk, "layer_columns", fake_columns)
    base = rdk.base_columns(runner, [], records)
    assert base[0][0] == [1, 2] and base[1][0] == [0]
    changed_by["on"] = True
    monkeypatch.setattr(rdk, "readings", lambda _r, rec, scale: {"passes": scale == 2.0})
    on_base = {r.submission.identifier.slug: {"passes": True} for r in records}
    change = rdk.change_for("instruct", rdk.deltas(base_model, other))
    summary, *rows = rdk.run_condition(
        runner, [], records, base, on_base, change, {"condition": "x"}
    )
    by_slug = {r["slug"]: r for r in rows}
    assert by_slug["L0-N1-positive"]["ratio"] == 2.0 and by_slug["L0-N1-positive"]["keeps"]
    assert by_slug["L0-N2-negative"]["ratio"] == 2.0
    # Neuron 0 of layer 1 has no positive peak on the base, so it has no ratio and scale 1.
    assert by_slug["L1-N0-positive"]["ratio"] is None and not by_slug["L1-N0-positive"]["keeps"]
    assert summary["keeps"] == 2 and summary["flipped"] == ["L1-N0-positive"]
    norm = float(torch.sqrt(sum((t.double() ** 2).sum() for t in change.values())))
    assert summary["change_norm"] == pytest.approx(norm, abs=1e-4)
