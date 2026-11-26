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


import json  # noqa: E402

import lora_units_stats as ls  # noqa: E402


def _record(layer, status, gates, *, lora=False, seed=None, active=0.01, spec=0.9):
    """One registry record of the study, with what the stats read."""
    ident = {"layer": layer}
    submission = {"identifier": ident, "provenance": {"selection": "manual"}}
    if lora:
        ident["superposition_id"] = f"s{layer}"
        submission["superposition_profile"] = {"method": "rank-one-lora"}
    else:
        ident.update(neuron_idx=layer * 10, pole="positive")
        submission["provenance"] = {"selection": "random", "seed": seed}
    metrics = {"specificity": spec, "sensitivity": 0.9, "balanced_accuracy": 0.9,
               "active_share": active, "gates": gates}  # fmt: skip
    return {"submission": submission,
            "verdict": {"status": status, "confidence_score": 0.8, "metrics": metrics}}  # fmt: skip


def _write(wiki, layer, name, stamp, record):
    folder = wiki / "experiments" / ls.MODEL_DIR / f"L{layer}" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stamp}.json").write_text(json.dumps(record))


def test_collect_keeps_the_study_and_the_newest_record_of_each_unit(tmp_path):
    ok = {"specificity": True, "sensitivity": True}
    _write(tmp_path, 2, "Ss2", "20261126T1", _record(2, "REJECTED", ok, lora=True))
    _write(tmp_path, 2, "Ss2", "20261126T2", _record(2, "VERIFIED", ok, lora=True))
    _write(tmp_path, 3, "N30-positive", "20261126T1", _record(3, "NEEDS_REVISION", ok, seed=5170))
    _write(tmp_path, 4, "N40-positive", "20261126T1", _record(4, "VERIFIED", ok, seed=1))
    units = ls.collect(tmp_path)
    assert [(u["arm"], u["unit"], u["status"]) for u in units] == [
        ("adapter", "s2", "VERIFIED"),
        ("native", "N30-positive", "NEEDS_REVISION"),
    ]
    assert units[1]["seed"] == 5170 and units[0]["failed"] == []


def test_unit_row_lists_the_failed_gates():
    record = _record(5, "NEEDS_REVISION", {"specificity": False, "sensitivity": True, "a": False})
    row = ls.unit_row(record, "native", "x.json")
    assert row["failed"] == ["a", "specificity"]
    assert row["unit"] == "N50-positive" and row["record"] == "x.json"


def _units():
    rows = []
    for i in range(4):
        rows.append({"arm": "adapter", "status": "VERIFIED" if i == 0 else "REJECTED",
                     "specificity": 0.2, "sensitivity": 1.0, "balanced_accuracy": 0.6,
                     "active_share": 0.05, "failed": ["specificity"] if i else []})  # fmt: skip
        rows.append({"arm": "native", "status": "VERIFIED" if i < 3 else "NEEDS_REVISION",
                     "specificity": 0.9, "sensitivity": 0.7, "balanced_accuracy": 0.8,
                     "active_share": None if i == 3 else 0.004, "failed": []})  # fmt: skip
    return rows


def test_arm_row_counts_verdicts_and_means_skip_missing_values():
    row = ls.arm_row(_units(), "native")
    assert (row["units"], row["verified"], row["needs_revision"], row["rejected"]) == (4, 3, 1, 0)
    assert row["verified_share"] == 0.75
    assert row["verified_interval"][0] < 0.75 < row["verified_interval"][1]
    assert row["mean_active_share"] == pytest.approx(0.004)


def test_test_rows_compare_the_arms():
    rows = {r["measure"]: r for r in ls.test_rows(_units())}
    assert rows["verified_share"]["test"] == "fisher_exact"
    assert 0 < rows["verified_share"]["p"] <= 1
    assert rows["specificity"]["difference"] == pytest.approx(-0.7)
    assert rows["active_share"]["difference"] == pytest.approx(0.046)


def test_gate_rows_count_each_arms_failures():
    assert ls.gate_rows(_units()) == [
        {"table": "gate", "gate": "specificity", "adapter": 3, "native": 0}
    ]


def test_permutation_p_is_seeded_and_at_least_its_floor():
    a, b = [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]
    assert ls.permutation_p(a, b, n=500, seed=1) == ls.permutation_p(a, b, n=500, seed=1)
    assert ls.permutation_p([0.0] * 10, [1.0] * 10, n=99) == pytest.approx(0.01)


def test_sign_p_is_symmetric():
    assert ls.sign_p(2, 10) == ls.sign_p(8, 10)
    assert ls.sign_p(0, 4) == pytest.approx(0.125)


def test_loading_row_skips_layers_with_no_verified_neuron():
    rows = [{"table": "loading", "verified": 1, "median_percentile": 0.9,
             "share_of_norm": 0.2, "share_expected": 0.1},
            {"table": "loading", "verified": 0, "median_percentile": None,
             "share_of_norm": None, "share_expected": 0.0},
            {"table": "unit"}]  # fmt: skip
    got = ls.loading_row(rows)
    assert (got["layers"], got["verified"], got["layers_above"], got["norm_ratio"]) == (
        1,
        1,
        1,
        2.0,
    )


def test_training_record_gives_the_steps_losses_and_cosines():
    adapters = lu.attach(_model(layers=1), seed=2)
    initial = {name: a.lora_A.detach().clone() for name, a in adapters.items()}
    got = lu.training_record(adapters, initial, fitted=17, before=1.234567, after=1.0)
    assert got["steps"] == lu.EPOCHS * (17 // lu.ACCUM)
    assert (got["held_out_loss_before"], got["held_out_loss_after"]) == (1.23457, 1.0)
    assert got["cos_to_init"] == {p: 1.0 for p in lu.PROJS}
    assert got["model"] == lu.MODEL and got["table"] == "training"
