"""``scripts/dormant_rank.py``: dormant poles and effective rank in training (#5134)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "dormant_rank", Path(__file__).parents[1] / "scripts" / "dormant_rank.py"
)
dr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dr)


def test_a_pole_is_active_by_the_sign_of_its_activation():
    acts = torch.tensor([[1.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
    assert dr.active_shares(acts).tolist() == [1.0, 0.0, 0.0, 0.0, 0.5, 0.0]


def test_the_line_is_a_share_of_each_poles_own_maximum():
    acts = torch.tensor([[1.0, -1.0], [0.1, 1.0], [0.3, 1.0]])
    shares = dr.line_shares(acts).tolist()
    assert shares[:2] == pytest.approx([2 / 3, 2 / 3])  # 0.25 and 1.0 reach their lines
    assert shares[2:] == pytest.approx([0.0, 1 / 3])  # the negative pole of neuron 0 is never on
    assert dr.line_shares(torch.tensor([[0.25], [1.0]])).tolist()[0] == 1.0  # at the line counts


def test_effective_rank_runs_from_one_to_the_rank():
    assert dr.effective_rank(torch.eye(4)) == pytest.approx(4.0)
    assert dr.effective_rank(torch.ones(3, 5)) == pytest.approx(1.0)
    assert dr.effective_rank(torch.zeros(3, 3)) == 0.0
    assert dr.effective_rank(torch.diag(torch.tensor([3.0, 1.0, 0.0]))) == pytest.approx(
        float(
            torch.exp(
                -(0.75 * torch.log(torch.tensor(0.75)) + 0.25 * torch.log(torch.tensor(0.25)))
            )
        )
    )


def test_the_token_sample_is_seeded_and_capped():
    assert dr.sample_rows(3, 5).tolist() == [0, 1, 2]
    assert dr.sample_rows(5, 5).tolist() == [0, 1, 2, 3, 4]
    rows = dr.sample_rows(100, 10)
    assert (
        len(rows) == 10 and len(set(rows.tolist())) == 10 and rows.tolist() == sorted(rows.tolist())
    )
    assert torch.equal(rows, dr.sample_rows(100, 10, seed=0))
    assert not torch.equal(rows, dr.sample_rows(100, 10, seed=1))
    assert len(dr.sample_rows(3000)) == dr.SAMPLE == 2000


def test_a_layer_row_counts_dormant_and_never_firing_poles():
    acts = torch.ones(200, 2)
    acts[0, 0] = -1.0  # neuron 0's negative pole is active on 0.5% of tokens: dormant
    acts[:, 1] = -1.0  # neuron 1 never fires positive and its negative pole is always on
    acts[:4, 1] = 2.0  # ... except on 2% of tokens, so its positive pole is not dormant
    row = dr.layer_row(512, 3, acts)
    assert row["table"] == "layer" and row["step"] == 512 and row["layer"] == 3
    assert row["tokens"] == 200 and row["poles"] == 4
    assert row["dormant"] == 0.25 and row["never"] == 0.0
    assert row["dormant_line"] == 0.25
    assert row["erank"] == pytest.approx(round(dr.effective_rank(acts), 3))


def test_a_pole_that_never_fires_counts_as_never_and_dormant():
    row = dr.layer_row(0, 0, torch.tensor([[1.0, 1.0], [2.0, 3.0]]))
    assert row["never"] == 0.5 and row["dormant"] == 0.5  # both negative poles


def test_the_rank_samples_the_tokens_and_also_uses_them_all(monkeypatch):
    seen = []
    monkeypatch.setattr(dr, "effective_rank", lambda m: seen.append(m.shape[0]) or 1.0)
    monkeypatch.setattr(dr, "SAMPLE", 3)
    monkeypatch.setattr(dr, "sample_rows", lambda n: torch.arange(3))
    dr.layer_row(0, 0, torch.ones(10, 2))
    assert seen == [3, 10]


def test_each_layer_stacks_the_windows_tokens():
    calls = []

    def layer_activations(texts, layers):
        calls.append((texts, layers))
        return {layer: torch.full((1, len(texts[0]), 2), float(layer)) for layer in layers}

    runner = SimpleNamespace(n_layers=2, layer_activations=layer_activations)
    acts = dr.step_activations(runner, ["abc", "de"])
    assert calls == [(["abc"], [0, 1]), (["de"], [0, 1])]  # one window at a time: no padding
    assert acts[0].shape == (5, 2) and acts[1].unique().tolist() == [1.0]
    assert acts[0].dtype == torch.float32
