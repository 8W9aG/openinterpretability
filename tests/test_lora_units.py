"""The helpers of scripts/lora_units.py (#5166)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import lora_units as lu  # noqa: E402


def _mlp(width=4, hidden=6):
    """One MLP with the three projections of a Qwen2 layer."""
    return SimpleNamespace(
        gate_proj=torch.nn.Linear(width, hidden, bias=False),
        up_proj=torch.nn.Linear(width, hidden, bias=False),
        down_proj=torch.nn.Linear(hidden, width, bias=False),
    )


def _model(layers=2):
    """A stand-in with ``model.model.layers[i].mlp``, as a Qwen2 model has."""
    return SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=_mlp()) for _ in range(layers)])
    )


def test_a_new_adapter_gives_the_base_output():
    torch.manual_seed(0)
    base = torch.nn.Linear(4, 3)
    adapter = lu.RankOne(base, torch.Generator().manual_seed(1))
    x = torch.randn(2, 5, 4)
    assert torch.equal(adapter(x), base(x))
    assert torch.count_nonzero(adapter.lora_B) == 0
    assert adapter.lora_A.shape == (4,) and adapter.lora_B.shape == (3,)


def test_the_adapter_starts_small_and_adds_its_rank_one_write():
    base = torch.nn.Linear(100, 2)
    adapter = lu.RankOne(base, torch.Generator().manual_seed(1))
    assert float(adapter.lora_A.detach().norm()) < 5 * lu.INIT
    with torch.no_grad():
        adapter.lora_A.copy_(torch.ones(100) / 100)
        adapter.lora_B.copy_(torch.tensor([2.0, -1.0]))
    x = torch.ones(1, 100)
    want = base(x) + lu.ALPHA * torch.tensor([[2.0, -1.0]])
    assert torch.allclose(adapter(x), want)


def test_attach_wraps_every_projection_of_every_layer():
    model = _model(layers=3)
    adapters = lu.attach(model, seed=4)
    assert sorted(adapters) == sorted(f"layers.{i}.{p}" for i in range(3) for p in lu.PROJS)
    for i, layer in enumerate(model.model.layers):
        for proj in lu.PROJS:
            assert getattr(layer.mlp, proj) is adapters[f"layers.{i}.{proj}"]
    # The adapters draw from one generator, so no two start the same.
    starts = [a.lora_A for a in adapters.values() if a.lora_A.shape == (4,)]
    assert not torch.equal(starts[0], starts[1])


def test_attach_is_seeded():
    one = lu.attach(_model(), seed=7)
    two = lu.attach(_model(), seed=7)
    other = lu.attach(_model(), seed=8)
    name = "layers.1.down_proj"
    assert torch.equal(one[name].lora_A, two[name].lora_A)
    assert not torch.equal(one[name].lora_A, other[name].lora_A)


def test_split_is_seeded_disjoint_and_sized():
    train, held = lu.split(1000, seed=3)
    assert (train, held) == lu.split(1000, seed=3)
    assert train != lu.split(1000, seed=4)[0]
    assert len(train) == lu.TRAIN and len(held) == lu.HELD_OUT
    assert not set(train) & set(held) and set(train) | set(held) <= set(range(1000))


def test_mean_loss_averages_the_batches_without_a_gradient():
    seen = []

    def model(batch, labels):
        seen.append((torch.is_grad_enabled(), torch.equal(batch, labels)))
        return SimpleNamespace(loss=torch.tensor(float(batch.sum())))

    got = lu.mean_loss(model, [torch.tensor([[1, 2]]), torch.tensor([[3, 6]])])
    assert got == pytest.approx(6.0)
    assert seen == [(False, True), (False, True)]


def test_cos_to_init_is_one_for_an_unchanged_direction_and_falls_as_it_turns():
    adapters = lu.attach(_model(layers=2), seed=1)
    initial = {name: a.lora_A.detach().clone() for name, a in adapters.items()}
    with torch.no_grad():
        for name, a in adapters.items():
            if name.endswith("down_proj"):
                a.lora_A.copy_(-initial[name])
    got = lu.cos_to_init(adapters, initial)
    assert set(got) == set(lu.PROJS)
    assert got["gate_proj"] == pytest.approx(1.0) and got["up_proj"] == pytest.approx(1.0)
    assert got["down_proj"] == pytest.approx(-1.0)


def test_superposition_keeps_the_sign_and_names_its_pole():
    unit = lu.superposition(5, torch.tensor([1.0, 0.0, -0.25]), 1)
    assert unit["members"] == [{"neuron_idx": 0, "weight": 1.0}, {"neuron_idx": 2, "weight": -0.25}]
    assert unit["model"] == lu.MODEL and unit["layer"] == 5 and unit["method"] == "rank-one-lora"
    assert "layer 5, positive side" in unit["source"]
    assert "negative side" in lu.superposition(5, torch.tensor([1.0]), -1)["source"]


def test_draw_is_seeded_and_never_repeats_a_pole():
    picks = lu.draw(3, n=6, seed=2)
    assert sorted(picks) == [(layer, s) for layer in range(3) for s in (-1, 1)]
    assert picks == lu.draw(3, n=6, seed=2)
    assert lu.draw(24, seed=2) != lu.draw(24, seed=9)


def test_loading_ranks_by_magnitude_whatever_the_sign():
    got = lu.loading(torch.tensor([-3.0, 0.1, 0.2, 2.0, -0.05]), [0, 4])
    assert got["verified"] == 2
    assert got["median_percentile"] == pytest.approx((1.0 + 0.0) / 2)
    assert got["share_of_norm"] == pytest.approx(
        round((9 + 0.0025) / (9 + 0.01 + 0.04 + 4 + 0.0025), 4)
    )
    assert got["share_expected"] == pytest.approx(0.4)


def test_loading_with_no_verified_neuron_has_no_percentile_or_share():
    got = lu.loading(torch.tensor([1.0, 2.0]), [])
    assert got == {
        "verified": 0,
        "median_percentile": None,
        "share_of_norm": None,
        "share_expected": 0.0,
    }


def test_texts_join_question_thinking_and_answer():
    row = {"question": "2+2?", "deepseek_thinking_trajectory": "Add.", "deepseek_attempt": "4"}
    assert lu.texts([row, row]) == ["Question: 2+2?\n\nThinking: Add.\n\nAnswer: 4"] * 2
