"""The diagonal Fisher of MLP neurons and its tests (interference/fisher.py, #5118)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.interference import fisher
from tests.conftest import FakeTokenizer


@pytest.fixture
def qwen_runner():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=16, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2)  # fmt: skip
    return NeuronHookRunner(Qwen2ForCausalLM(config).eval(), FakeTokenizer())


def test_neuron_weights_finds_both_input_rows_and_the_output_column(qwen_runner):
    found = fisher.neuron_weights(qwen_runner, 0)
    assert [(side, tuple(w.shape), axis) for side, w, axis in found] == [
        ("input", (48, 32), 0), ("input", (48, 32), 0), ("output", (32, 48), 1)]  # fmt: skip


def test_neuron_weights_reads_conv1d_axes(tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    found = fisher.neuron_weights(runner, 0)
    width = runner.mlp_width(0)
    assert [(side, axis) for side, _, axis in found] == [("input", 1), ("output", 0)]
    assert all(w.shape[axis] == width for _, w, axis in found)


def test_fisher_windows_cuts_rows_and_refuses_a_short_sample():
    rows = fisher.fisher_windows([torch.arange(10).view(2, 5), torch.arange(10, 13).view(1, 3)], 4)
    assert [r.tolist() for r in rows] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    with pytest.raises(ValueError, match="less than one window"):
        fisher.fisher_windows([torch.arange(3).view(1, 3)], 4)


def test_diagonal_fisher_matches_a_hand_computed_squared_gradient(qwen_runner):
    rows = [torch.randint(0, 16, (6,)) for _ in range(2)]
    before = {n: p.requires_grad for n, p in qwen_runner.model.named_parameters()}
    out = fisher.diagonal_fisher(qwen_runner, rows, seed=1)
    assert {n: p.requires_grad for n, p in qwen_runner.model.named_parameters()} == before
    assert set(out) == set(fisher.LABELS)
    assert out["data"]["input"].shape == (2, 48)
    # The empirical Fisher of layer 1's output column, by hand.
    down = qwen_runner.down_proj(1).weight
    want = torch.zeros(48)
    for row in rows:
        logits = qwen_runner.model(row.view(1, -1)).logits[0, :-1]
        loglik = torch.log_softmax(logits, -1).gather(1, row[1:].unsqueeze(1)).sum()
        (grad,) = torch.autograd.grad(loglik, [down])
        want += grad.pow(2).sum(0)
    assert torch.allclose(out["data"]["output"][1], want / 2, rtol=1e-4)
    assert (out["model"]["input"] > 0).all()


def test_layer_percentile_counts_ties_half():
    values = torch.tensor([[1.0, 2.0, 2.0, 3.0]])
    assert fisher.layer_percentile(values, 0, 0) == 0.0
    assert fisher.layer_percentile(values, 0, 1) == pytest.approx(1.5 / 3)
    assert fisher.layer_percentile(values, 0, 3) == 1.0


def test_auc_matches_the_pairwise_count_and_refuses_one_label():
    scores, labels = [0.1, 0.4, 0.4, 0.9, 0.3], [False, True, False, True, True]
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    pairs = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert fisher.auc(scores, labels) == pytest.approx(pairs)
    with pytest.raises(ValueError, match="both labels"):
        fisher.auc([1, 2], [True, True])


def test_spearman_reads_ranks_and_a_constant_side():
    assert fisher.spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == pytest.approx(1.0)
    assert fisher.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert fisher.spearman([1, 2, 3], [5, 5, 5]) is None


def test_stratified_null_shuffles_only_within_strata():
    # Each stratum holds one outcome only, so no shuffle can change the statistic.
    scores = [0.1, 0.2, 0.8, 0.9]
    out = fisher.stratified_null(lambda ys: fisher.auc(scores, ys),
                                 [False, False, True, True], [0, 0, 1, 1], n=50)  # fmt: skip
    assert out["observed"] == 1.0 and set(out["null"]) == {1.0} and out["p"] == 1.0


def test_stratified_null_finds_a_strong_effect():
    scores = [i / 40 for i in range(40)]
    ys = [i >= 20 for i in range(40)]
    out = fisher.stratified_null(lambda y: fisher.auc(scores, y), ys, [0] * 40, n=200)
    assert out["observed"] == 1.0
    assert out["p"] == pytest.approx(1 / 201)
    assert abs(out["null_mean"] - 0.5) < 0.05


def _script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "fisher_keep.py"
    spec = importlib.util.spec_from_file_location("fisher_keep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_reads_poles_adds_fisher_and_runs_the_tests(tmp_path, monkeypatch):
    script = _script()
    monkeypatch.setattr(script, "PERMUTATIONS", 20)
    monkeypatch.setattr(script, "stratified_null",
                        lambda *a, **k: fisher.stratified_null(*a, n=20, **k))  # fmt: skip

    def pole(n, layer, keeps, passes=True):
        base = {"specificity": 0.9, "sensitivity": 0.7, "passes": passes}
        return {"table": "pole", "slug": f"M/L{layer}/N{n}-positive", "layer": layer,
                "pole": "positive", "depth": layer, "keeps": keeps, "base": base,
                "pearson": 0.9 + n / 100, "ratio": 1 + n / 10}  # fmt: skip

    plain = [pole(n, n % 2, n > 1) for n in range(4)] + [pole(5, 0, False, passes=False)]
    chat = [pole(n, n % 2, n > 1) for n in range(3)]
    (tmp_path / "p.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [{"table": "summary"}, *plain])
    )
    (tmp_path / "c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chat))
    poles = script.load_poles(tmp_path / "p.jsonl", tmp_path / "c.jsonl")
    assert [p["neuron"] for p in poles] == [0, 1, 2, 3]  # the pole that didn't reproduce is out
    assert poles[3]["keeps_chat"] is None and poles[0]["margin"] == pytest.approx(0.1)
    values = torch.arange(2 * 4, dtype=torch.float).view(2, 4)
    rms = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 5.0]])
    script.add_fisher(poles, {k: {"input": values, "output": values} for k in fisher.LABELS}, rms)
    assert poles[2]["percentile"]["model"]["total"] == pytest.approx(2 / 3)
    assert poles[0]["percentile"]["rms"] == 1.0 and poles[0]["rms"] == 4.0
    rows = script.tests(poles)
    assert {r["outcome"] for r in rows} == {
        "keeps",
        "keeps_chat",
        "one_minus_r",
        "abs_log_ratio",
        "margin",
    }
    primary = [r for r in rows if "null" in r]
    assert len(primary) == 1 and len(primary[0]["null"]) == 20
    assert all(0 < r["p"] <= 1 and not math.isnan(r["observed"]) for r in rows)
    assert {r["statistic"] for r in rows} == {"auc", "spearman", "partial_spearman"}
    assert any(r["fisher"] == "rms" and r["outcome"] == "keeps" for r in rows)
    assert not any(r["fisher"] == "rms" and r["statistic"] == "partial_spearman" for r in rows)


def test_script_neuron_of_refuses_a_slug_with_no_index():
    with pytest.raises(ValueError, match="no neuron index"):
        _script().neuron_of("M/L0/attn")


def test_activation_rms_matches_the_down_projections_input(qwen_runner):
    rows = [torch.randint(0, 16, (5,)) for _ in range(2)]
    seen = []
    handle = qwen_runner.down_proj(1).register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
    try:
        rms = fisher.activation_rms(qwen_runner, rows)
    finally:
        handle.remove()
    want = torch.cat([a.reshape(-1, 48) for a in seen]).pow(2).mean(0).sqrt()
    assert rms.shape == (2, 48) and torch.allclose(rms[1], want, rtol=1e-5)


def test_partial_spearman_removes_a_shared_control():
    control = list(range(20))
    a = [c + (i * 7) % 11 for i, c in enumerate(control)]  # the control plus unrelated noise
    b = [c + (i * 5) % 13 for i, c in enumerate(control)]
    plain, partial = fisher.spearman(a, b), fisher.partial_spearman(a, b, control)
    assert plain > 0.6 and partial is not None and abs(partial) < plain - 0.3
    assert fisher.partial_spearman(control, b, control) is None
    # A constant control leaves the plain Spearman.
    assert fisher.partial_spearman([1, 2, 3], [1, 3, 2], [0, 0, 0]) == pytest.approx(0.5)


def test_neuron_weights_skips_a_matrix_that_is_not_one_neuron_per_slice():
    from types import SimpleNamespace

    from torch import nn

    up, down, other = nn.Linear(4, 6), nn.Linear(6, 4), nn.Linear(4, 3)
    mlp = nn.ModuleDict({"up": up, "down": down, "other": other})
    runner = SimpleNamespace(mlp_width=lambda _l: 6, down_proj=lambda _l: down, mlp=lambda _l: mlp)
    found = fisher.neuron_weights(runner, 0)
    assert [(side, w is up.weight or w is down.weight) for side, w, _ in found] == [
        ("input", True),
        ("output", True),
    ]
