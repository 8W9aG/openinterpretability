"""Pruning a neuron's input weights, and the pruning keep script (#5122)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.interference import pruning
from tests.conftest import FakeTokenizer


@pytest.fixture
def neox_runner():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    config = GPTNeoXConfig(vocab_size=16, hidden_size=32, intermediate_size=48,
                           num_hidden_layers=2, num_attention_heads=4)  # fmt: skip
    return NeuronHookRunner(GPTNeoXForCausalLM(config).eval(), FakeTokenizer())


@pytest.fixture
def qwen_runner():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=16, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2)  # fmt: skip
    return NeuronHookRunner(Qwen2ForCausalLM(config).eval(), FakeTokenizer())


def test_zeroed_takes_the_smallest_first_or_a_seeded_draw():
    row = torch.tensor([0.4, -0.1, 3.0, -2.0, 0.2, 1.0, 0.05, -0.3])
    assert pruning.zeroed(row, 0.5, "magnitude").nonzero().flatten().tolist() == [1, 4, 6, 7]
    draw = lambda: pruning.zeroed(row, 0.75, "random", torch.Generator().manual_seed(3))  # noqa: E731
    assert draw().sum() == 6 and torch.equal(draw(), draw())
    with pytest.raises(ValueError, match="unknown method"):
        pruning.zeroed(row, 0.5, "largest")


def test_pruned_zeroes_each_input_row_and_restores_it(neox_runner):
    weight = neox_runner.mlp(1).dense_h_to_4h.weight
    before = weight.detach().clone()
    other = neox_runner.mlp(1).dense_4h_to_h.weight.detach().clone()
    with pruning.pruned(neox_runner, 1, [3, 7], 0.75, "magnitude"):
        assert (weight[3] == 0).sum() == 24 and (weight[7] == 0).sum() == 24
        assert torch.equal(weight[0], before[0])  # another neuron is untouched
        kept = weight[3][weight[3] != 0].abs().min()
        assert kept >= before[3].abs().sort().values[23]  # the largest quarter stays
    assert torch.equal(weight, before)
    assert torch.equal(neox_runner.mlp(1).dense_4h_to_h.weight, other)  # the output side stays


def test_pruned_restores_after_an_error_and_prunes_both_qwen_rows(qwen_runner):
    gate, up = qwen_runner.mlp(0).gate_proj.weight, qwen_runner.mlp(0).up_proj.weight
    before = (gate.detach().clone(), up.detach().clone())
    with pytest.raises(RuntimeError), pruning.pruned(qwen_runner, 0, [5], 0.5, "random"):
        assert (gate[5] == 0).sum() == 16 and (up[5] == 0).sum() == 16
        raise RuntimeError("stop")
    assert torch.equal(gate, before[0]) and torch.equal(up, before[1])


def test_pruning_one_neuron_leaves_its_layer_neighbours_alone(neox_runner):
    batch = {"input_ids": torch.randint(0, 16, (1, 6))}
    acts = neox_runner.layer_activations(batch, [1])[1]
    with pruning.pruned(neox_runner, 1, [3], 0.9, "magnitude"):
        after = neox_runner.layer_activations(batch, [1])[1]
    assert not torch.equal(acts[..., 3], after[..., 3])
    assert torch.equal(acts[..., :3], after[..., :3]) and torch.equal(acts[..., 4:], after[..., 4:])


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pruning_keep.py"
    spec = importlib.util.spec_from_file_location("pruning_keep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(layer, neuron, pole="positive"):
    ident = SimpleNamespace(layer=layer, neuron_idx=neuron, pole=pole,
                            slug=f"M/L{layer}/N{neuron}-{pole}")  # fmt: skip
    return SimpleNamespace(submission=SimpleNamespace(identifier=ident))


def test_script_reads_each_variant_against_the_unpruned_columns(monkeypatch, neox_runner):
    script = _script()
    keeps = iter([True, True, True, False])
    monkeypatch.setattr(script, "readings", lambda runner, record, ratio: {
        "specificity": 1.0, "sensitivity": 0.9, "failed": [], "passes": next(keeps)})  # fmt: skip
    batches = [torch.randint(0, 16, (2, 8))]
    records = [_record(1, 3), _record(1, 3, "negative")]
    base: dict = {}
    rows = script.measure(neox_runner, batches, records, None, base)
    assert [r["method"] for r in rows] == ["none", "none"] and rows[0]["pearson"] == 1.0
    rows += script.measure(neox_runner, batches, records, (0.9, "magnitude"), base)
    assert rows[2]["fraction"] == 0.9 and rows[2]["pearson"] < 1.0
    assert set(base) == {1} and not rows[3]["keeps"]
    assert list(script.by_layer([_record(2, 1), _record(0, 4)])) == [0, 2]


def test_summary_counts_only_poles_that_reproduce():
    script = _script()
    rows = [{"slug": s, "method": "none", "fraction": 0.0, "keeps": k, "pearson": 1.0}
            for s, k in (("a", True), ("b", True), ("c", False))]  # fmt: skip
    for fraction in pruning.LEVELS:
        for method in pruning.METHODS:
            rows += [{"slug": s, "method": method, "fraction": fraction, "pearson": 0.5,
                      "keeps": s == "a" or method == "random"} for s in "abc"]  # fmt: skip
    levels = script.summary(rows)
    first = levels[0]
    assert (first["fraction"], first["method"], first["poles"], first["keeps"]) == (
        0.5, "magnitude", 2, 1)  # fmt: skip
    assert levels[1]["keeps"] == 2 and first["median_pearson"] == 0.5
