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


import wrapper_prune_check as check  # noqa: E402
import wrapper_prune_stats as stats  # noqa: E402


def _pole(slug, condition, k, seed, keeps, base_passes=True):
    return {"table": "pole", "slug": slug, "condition": condition, "k": k, "seed": seed,
            "keeps": keeps, "base_passes": base_passes}  # fmt: skip


def test_changes_count_recovered_and_lost_poles_against_the_instruct_model():
    rows = [
        _pole("a", "none", 0, None, False), _pole("b", "none", 0, None, False),
        _pole("c", "none", 0, None, True), _pole("d", "none", 0, None, True),
        _pole("e", "none", 0, None, False, base_passes=False),
        _pole("a", "top", 5, None, True), _pole("b", "top", 5, None, True),
        _pole("c", "top", 5, None, False), _pole("d", "top", 5, None, True),
        _pole("a", "random", 5, 0, False), _pole("b", "random", 5, 0, True),
        _pole("c", "random", 5, 0, True), _pole("d", "random", 5, 0, True),
    ] + [{"table": "summary", "condition": c, "k": k, "seed": s, "held_kl": kl,
          "neurons": n}
         for c, k, s, kl, n in (("none", 0, None, 0.1, []), ("top", 5, None, 0.3, [(1, 2)]),
                                ("random", 5, 0, 0.1, [(1, 7)]))]  # fmt: skip
    top, rand = stats.changes(rows)
    assert (top["recovered"], top["lost"], top["held_kl"]) == (2, 1, 0.3)
    assert (top["failed_on_instruct"], top["kept_on_instruct"]) == (2, 2)
    assert "recovered_p" not in top
    assert (rand["recovered"], rand["lost"]) == (1, 0)
    assert rand["recovered_p"] == 1.0 and rand["lost_p"] == 1.0
    assert stats.layers_of_top(rows) == [{"table": "top_layers", "k": 5, "layers": [(1, 1)]}]


def test_ranked_orders_the_scores_highest_first():
    scores = {"0": [0.1, 0.5], "3": [0.9, -1.0]}
    assert check.ranked(scores, 2) == [(3, 0, 0.9), (0, 1, 0.5)]


class _GatedMlp(nn.Module):
    """Qwen2's MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, width=3, hidden=4):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def _gated(layers=2):
    model = _Tiny(layers)
    for layer in model.model.layers:
        layer.mlp = _GatedMlp()
    return model


def test_an_input_score_predicts_the_kl_drop_of_zeroing_the_input_rows():
    torch.manual_seed(1)
    model, windows = _gated(), [torch.tensor([[0, 1, 2, 3]]), torch.tensor([[4, 2, 1, 0]])]
    base = [torch.log_softmax(torch.randn(4, 5), dim=-1).half() for _ in windows]
    got = wp.scores(model, windows, base, "cpu", unit="input")
    assert set(got) == {0, 1} and got[0].shape == (4,)
    assert all(p.grad is None for p in model.parameters())
    assert not torch.allclose(got[1], wp.scores(model, windows, base, "cpu")[1])
    before = wp.mean_kl(model, windows, base, "cpu")
    mlp = model.model.layers[1].mlp
    saved = [w.detach().clone() for w in (mlp.gate_proj.weight, mlp.up_proj.weight)]
    with torch.no_grad():
        mlp.gate_proj.weight[1] = 0
        mlp.up_proj.weight[1] = 0
    drop = before - wp.mean_kl(model, windows, base, "cpu")
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(saved[0])
        mlp.up_proj.weight.copy_(saved[1])
    # Second-order terms are larger here, as two rows go at once.
    assert drop == pytest.approx(float(got[1][1]), abs=0.25 * abs(float(got[1][1])) + 2e-3)


def test_an_input_prune_zeroes_only_the_chosen_rows_and_switches_the_neuron_off():
    torch.manual_seed(0)
    base, instruct = _gated(), _gated()
    delta = {n: (dict(instruct.named_parameters())[n] - p).detach()
             for n, p in base.named_parameters()}  # fmt: skip
    change = wp.pruned_change(delta, base, [(0, 2)], unit="input")
    params = dict(base.named_parameters())
    for name in wp.input_names(0):
        pruned = params[name].detach() + change[name]
        want = dict(instruct.named_parameters())[name].detach().clone()
        want[2] = 0
        assert torch.allclose(pruned, want)
    untouched = set(delta) - set(wp.input_names(0))
    assert all(torch.equal(change[n], delta[n]) for n in untouched)
    assert wp.input_names(3) == ("model.layers.3.mlp.gate_proj.weight",
                                 "model.layers.3.mlp.up_proj.weight")  # fmt: skip
    mlp = _GatedMlp()
    with torch.no_grad():
        mlp.gate_proj.weight[2] = 0
        mlp.up_proj.weight[2] = 0
    x = torch.randn(5, 3)
    hidden = nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
    assert torch.equal(hidden[:, 2], torch.zeros(5))


def test_the_split_is_seeded_disjoint_and_covers_every_failed_pole():
    slugs = [f"m/L0/N{i}-positive" for i in range(7)]
    scored, held = wp.split_failed(slugs)
    assert len(scored) == 4 and len(held) == 3
    assert sorted(scored + held) == sorted(slugs) and not set(scored) & set(held)
    assert (scored, held) == wp.split_failed(list(reversed(slugs)))
    assert (scored, held) != wp.split_failed(slugs, seed=1)
    assert wp.split_failed([]) == ([], [])


def test_failed_slugs_take_the_poles_that_pass_on_base_and_fail_on_instruct():
    rows = [{"table": "summary"}, _pole("a", "none", 0, None, False),
            _pole("b", "none", 0, None, True),
            _pole("c", "none", 0, None, False, base_passes=False)]  # fmt: skip
    assert wp.failed_slugs(rows) == ["a"]


class _Record:
    def __init__(self, texts):
        kinds = ("hard_negative", "positive_holdout", "minimal_pair")[: len(texts)]
        results = [
            type("R", (), {"prompt": t, "kind": kind})()
            for t, kind in zip(texts, kinds, strict=True)
        ]
        self.verdict = type("V", (), {"adversarial_test_results": results})()


def test_text_windows_tokenize_each_probe_text_as_one_row():
    tokenizer = lambda text: {"input_ids": [len(w) for w in text.split()]}  # noqa: E731
    got = wp.text_windows(tokenizer, [_Record(["ab c", "abc", "skipped pair"]), _Record(["a"])])
    assert [w.tolist() for w in got] == [[[2, 1]], [[3]], [[1]]]


def _summary(condition, k, seed, neurons, **variant):
    return {"table": "summary", "condition": condition, "k": k, "seed": seed, "held_kl": 0.1,
            "neurons": neurons, **variant}  # fmt: skip


def test_an_input_variant_leaves_out_the_poles_its_prune_switches_off():
    v = {"unit": "input", "loss": "base"}
    a, b, c = "m/L1/N2-positive", "m/L1/N3-negative", "m/L0/N5-positive"
    rows = [
        {**_pole(a, "none", 0, None, False), **v}, {**_pole(b, "none", 0, None, True), **v},
        {**_pole(c, "none", 0, None, False), **v},
        {**_pole(a, "top", 5, None, False), **v}, {**_pole(b, "top", 5, None, False), **v},
        {**_pole(c, "top", 5, None, True), **v},
        {**_pole(a, "random", 5, 0, False), **v}, {**_pole(b, "random", 5, 0, True), **v},
        {**_pole(c, "random", 5, 0, False), **v},
        _summary("none", 0, None, [], **v), _summary("top", 5, None, [[1, 2]], **v),
        _summary("random", 5, 0, [[1, 3]], **v),
        # The paper's own rows, with no variant fields, are ignored.
        _pole(a, "none", 0, None, True), _summary("none", 0, None, []),
    ]  # fmt: skip
    top, rand = stats.changes(rows, unit="input")
    assert (top["switched_off"], top["failed_on_instruct"], top["recovered"]) == (1, 1, 1)
    assert (top["kept_on_instruct"], top["lost"]) == (1, 1)
    assert (rand["switched_off"], rand["kept_on_instruct"], rand["lost"]) == (1, 0, 0)
    assert rand["recovered_p"] == 1.0 and rand["lost_p"] == 1.0
    assert "held_recovered" not in top
    assert stats.neuron_of("Qwen--Qwen2.5-0.5B/L12/N340-negative") == (12, 340)
    assert stats.layers_of_top(rows, unit="input") == [
        {"table": "top_layers", "k": 5, "layers": [(1, 1)]}
    ]
    assert stats.layers_of_top(rows) == []


def test_a_failed_loss_variant_counts_the_recovered_poles_in_each_half():
    v = {"unit": "write", "loss": "failed"}
    names = ["s1", "s2", "h1", "h2", "k"]
    keeps_none = [False, False, False, False, True]
    keeps_top = [True, True, True, False, True]
    keeps_rand = [False, True, False, False, True]
    rows = [{**_pole(n, "none", 0, None, x), **v} for n, x in zip(names, keeps_none, strict=True)]
    rows += [{**_pole(n, "top", 5, None, x), **v} for n, x in zip(names, keeps_top, strict=True)]
    rows += [{**_pole(n, "random", 5, 0, x), **v} for n, x in zip(names, keeps_rand, strict=True)]
    rows += [_summary("none", 0, None, [], **v), _summary("top", 5, None, [], **v),
             _summary("random", 5, 0, [], **v),
             {"table": "split", **v, "seed": 0, "scored": ["s1", "s2"],
              "held": ["h1", "h2"]}]  # fmt: skip
    top, rand = stats.changes(rows, loss="failed")
    assert (
        top["recovered"],
        top["held_failed"],
        top["held_recovered"],
        top["scored_recovered"],
    ) == (3, 2, 1, 2)
    assert (rand["held_recovered"], rand["scored_recovered"]) == (0, 1)
    assert rand["held_recovered_p"] == 1.0
    assert "switched_off" not in top


def test_the_check_reads_the_paper_scores_not_a_variant():
    rows = [{"table": "scores", "unit": "input", "loss": "base", "scores": {"0": [9.0]}},
            {"table": "scores", "scores": {"0": [1.0]}},
            {"table": "scores", "unit": "write", "loss": "failed",
             "scores": {"0": [5.0]}}]  # fmt: skip
    assert check.paper_scores(rows) == {"0": [1.0]}
