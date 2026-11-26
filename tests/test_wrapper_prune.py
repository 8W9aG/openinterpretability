"""The helpers of scripts/wrapper_prune.py (#5163)."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import wrapper_prune as wp  # noqa: E402


def test_token_kl_is_zero_for_the_same_distribution_and_positive_otherwise():
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]])
    logp = torch.log_softmax(logits, dim=-1)
    assert float(wp.token_kl(logp, logits)) == pytest.approx(0.0, abs=1e-6)
    other = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p, q = logp.exp(), torch.softmax(other, dim=-1)
    expected = (p * (p.log() - q.log())).sum(-1).mean()
    assert float(wp.token_kl(logp, other)) == pytest.approx(float(expected), rel=1e-5)


class _Mlp(nn.Module):
    def __init__(self, width=3, hidden=4):
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up(x)))


class _Tiny(nn.Module):
    """A model with Qwen2's layer layout, and logits as its output."""

    def __init__(self, layers=2, vocab=5, width=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([type("L", (nn.Module,), {})() for _ in range(layers)])
        for layer in self.model.layers:
            layer.mlp = _Mlp(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        for layer in self.model.layers:
            h = h + layer.mlp(h)
        return type("O", (), {"logits": self.head(h)})()


def test_a_score_predicts_the_kl_drop_of_zeroing_a_column():
    torch.manual_seed(0)
    model, windows = _Tiny(), [torch.tensor([[0, 1, 2, 3]]), torch.tensor([[4, 2, 1, 0]])]
    base = [torch.log_softmax(torch.randn(4, 5), dim=-1).half() for _ in windows]
    got = wp.scores(model, windows, base, "cpu")
    assert set(got) == {0, 1} and got[0].shape == (4,)
    assert all(p.grad is None for p in model.parameters())  # the gradients are cleared
    before = wp.mean_kl(model, windows, base, "cpu")
    w = model.model.layers[1].mlp.down_proj.weight
    saved = w.detach().clone()
    with torch.no_grad():
        w[:, 2] = 0
    drop = before - wp.mean_kl(model, windows, base, "cpu")
    with torch.no_grad():
        w.copy_(saved)
    # First order: zeroing the column lowers the KL by about its score.
    assert drop == pytest.approx(float(got[1][2]), abs=0.05 * abs(float(got[1][2])) + 1e-3)


def test_top_neurons_take_the_highest_scores_over_every_layer():
    layer_scores = {0: torch.tensor([0.1, 0.9, -1.0]), 1: torch.tensor([0.5, 0.95, 0.0])}
    assert wp.top_neurons(layer_scores, 2) == [(0, 1), (1, 1)]
    assert wp.top_neurons(layer_scores, 3) == [(0, 1), (1, 0), (1, 1)]


def test_the_random_draw_matches_the_layer_counts_and_avoids_the_chosen():
    chosen = [(0, 1), (0, 4), (2, 0)]
    got = wp.random_like(chosen, {0: 10, 1: 10, 2: 10}, seed=3)
    assert sorted(layer for layer, _ in got) == [0, 0, 2]
    assert not set(got) & set(chosen)
    assert got == wp.random_like(chosen, {0: 10, 1: 10, 2: 10}, seed=3)
    assert got != wp.random_like(chosen, {0: 10, 1: 10, 2: 10}, seed=4)


def test_pruning_zeroes_only_the_chosen_columns_of_the_instruct_model():
    torch.manual_seed(0)
    base, instruct = _Tiny(), _Tiny()
    delta = {n: (dict(instruct.named_parameters())[n] - p).detach()
             for n, p in base.named_parameters()}  # fmt: skip
    change = wp.pruned_change(delta, base, [(1, 0), (1, 3)])
    name = "model.layers.1.mlp.down_proj.weight"
    pruned = dict(base.named_parameters())[name].detach() + change[name]
    want = dict(instruct.named_parameters())[name].detach().clone()
    want[:, [0, 3]] = 0
    assert torch.allclose(pruned, want)
    for other in delta:
        if other != name:
            assert torch.equal(change[other], delta[other])
    assert not torch.equal(delta[name][:, 0], change[name][:, 0])  # delta itself is untouched
    assert wp.down_name(4) == "model.layers.4.mlp.down_proj.weight"


def test_the_plan_runs_the_instruct_model_then_top_and_random_at_each_k():
    layer_scores = {0: torch.arange(600.0), 1: -torch.arange(600.0)}
    plan = wp.prune_plan(layer_scores, seeds=2)
    assert plan[0] == ("none", 0, None, [])
    assert len(plan) == 1 + len(wp.KS) * 3
    top5 = next(p for p in plan if p[:2] == ("top", 5))
    assert top5[3] == [(0, n) for n in range(595, 600)]
    randoms = [p for p in plan if p[:2] == ("random", 5)]
    assert [p[2] for p in randoms] == [0, 1] and all(len(p[3]) == 5 for p in randoms)
