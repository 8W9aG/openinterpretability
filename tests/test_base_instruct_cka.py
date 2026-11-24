"""Layer-level CKA and block swaps of base and instruct (scripts/base_instruct_cka.py, #5127)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import base_instruct_cka as cka  # noqa: E402

from openinterpretability.harness.hooks import NeuronHookRunner  # noqa: E402
from tests.conftest import VOCAB  # noqa: E402


def _runner(model):
    return NeuronHookRunner(model)


def _record(layer: int, slug: str):
    ident = SimpleNamespace(layer=layer, slug=slug)
    return SimpleNamespace(submission=SimpleNamespace(identifier=ident))


def test_cka_is_one_for_a_rotated_copy_and_lower_for_noise():
    x = torch.randn(200, 6, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(200, 6, generator=torch.Generator().manual_seed(2))
    assert cka.linear_cka(x, x) == 1.0
    assert cka.linear_cka(x, noise) < 0.2


def test_a_model_against_itself_has_cka_one_and_its_poles_correlate(tiny_gpt2):
    runner = _runner(tiny_gpt2)
    batches = [torch.randint(VOCAB, (2, 16), generator=torch.Generator().manual_seed(0))]
    stats = {"a": {"pearson": 1.0}, "b": {"pearson": None}}
    rows = cka.layer_rows(runner, runner, runner, batches, stats, {0: ["a", "b"]})
    assert [r["cka"] for r in rows] == [r["cka_standardised"] for r in rows] == [1.0, 1.0]
    assert rows[0]["poles"] == 1 and rows[0]["median_pole_pearson"] == 1.0
    assert rows[1]["poles"] == 0 and rows[1]["median_pole_pearson"] is None


def test_a_swap_copies_the_blocks_and_the_row_reads_each_third(tiny_gpt2):
    from transformers import GPT2LMHeadModel

    torch.manual_seed(1)
    other = GPT2LMHeadModel(tiny_gpt2.config).eval()
    target, source = _runner(other), _runner(tiny_gpt2)
    cka.swap_blocks(target, source, [1])
    same = [
        torch.equal(a, b)
        for a, b in zip(target.layers[1].parameters(), source.layers[1].parameters(), strict=True)
    ]
    assert all(same)
    assert not torch.equal(target.layers[0].mlp.c_fc.weight, source.layers[0].mlp.c_fc.weight)
    records = [_record(0, "x"), _record(2, "y"), _record(2, "z")]
    stats = {"x": {"pearson": 0.9}, "y": {"pearson": 0.5}, "z": {"pearson": None}}
    row = cka.swap_row([1], stats, records, n_layers=3)
    assert row["median_pole_pearson_by_third"] == [0.9, None, 0.5]
    assert row["pole_pearson"] == {"x": 0.9, "y": 0.5, "z": None}


def test_a_swap_is_undone_and_a_lost_copy_raises(tiny_gpt2, monkeypatch):
    import pytest
    from transformers import GPT2LMHeadModel

    torch.manual_seed(1)
    target, source = _runner(GPT2LMHeadModel(tiny_gpt2.config).eval()), _runner(tiny_gpt2)
    before = target.layers[0].mlp.c_fc.weight.detach().clone()
    cka.restore_blocks(target, cka.swap_blocks(target, source, [0]))
    assert torch.equal(target.layers[0].mlp.c_fc.weight, before)
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", lambda self, state: None)
    with pytest.raises(RuntimeError, match="didn't take the copy"):
        cka.swap_blocks(target, source, [0])
