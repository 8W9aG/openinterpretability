"""The helpers of scripts/traffic_map.py and scripts/traffic_stats.py (#5231)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import traffic_map as tm  # noqa: E402
import traffic_stats as ts  # noqa: E402


def _tiny():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    config = GPTNeoXConfig(vocab_size=20, hidden_size=16, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=32,
                           max_position_embeddings=16)  # fmt: skip
    return GPTNeoXForCausalLM(config).eval()


def test_every_neuron_write_and_head_is_a_site():
    model = _tiny()
    sites = tm.site_params(model)
    assert len(sites) == 2 * (32 + 4)
    kind, layer, index, name, cols = sites[32]  # the first head of layer 0
    assert (kind, layer, index) == ("head", 0, 0) and cols == slice(0, 4)
    assert name == "gpt_neox.layers.0.attention.dense.weight"
    assert sites[0][3] == "gpt_neox.layers.0.mlp.dense_4h_to_h.weight"


def test_column_norms_by_neuron_and_by_head():
    grad = torch.arange(24.0).reshape(3, 8)
    assert torch.allclose(tm.column_norms(grad, "mlp", 0), grad.norm(dim=0))
    heads = tm.column_norms(grad, "head", 2)
    assert torch.allclose(heads, torch.stack([grad[:, :4].norm(), grad[:, 4:].norm()]))


def test_the_score_reaches_only_writes_below_the_pole():
    model = _tiny()
    for p in model.parameters():
        p.requires_grad_(False)
    weights = tm.output_weights(model)
    for w in weights.values():
        w.requires_grad_(True)
    torch.set_grad_enabled(True)
    score = tm.peak_score(model, [[1, 2, 3], [4, 5]], 1, 3, "positive", False)
    score.backward()
    assert weights["mlp", 0].grad is not None and weights["head", 0].grad is not None
    assert weights["mlp", 1].grad is None  # the pole's own layer's write doesn't reach it
    assert tm.peak_score(model, [[1]], 1, 3, "positive", True) is None  # no token after the first


def test_a_change_has_the_norm_and_touches_only_the_site():
    model = _tiny()
    row = {"param": "gpt_neox.layers.1.attention.dense.weight", "start": 4, "stop": 8}
    change = tm.site_change(model, row, 2.5, seed=1)[row["param"]]
    assert float(change.norm()) == pytest.approx(2.5)
    assert float(change[:, :4].abs().sum()) == 0 and float(change[:, 8:].abs().sum()) == 0
    again = tm.site_change(model, row, 2.5, seed=1)[row["param"]]
    assert torch.equal(change, again)


def test_sites_are_sampled_by_decile_of_traffic_with_every_head_then_the_top():
    rows = [{"kind": "mlp", "traffic": float(i)} for i in range(100)]
    rows += [{"kind": "head", "traffic": 0.0} for _ in range(3)]
    got = tm.sample_sites(rows, per_decile=2, seed=1, top=4)
    draws = [d for d, _ in got]
    assert draws == ["decile"] * 20 + ["head"] * 3 + ["top"] * 4
    deciles = [int(r["traffic"]) // 10 for d, r in got if d == "decile"]
    assert deciles == sorted(deciles) and set(deciles) == set(range(10))
    tops = [r["traffic"] for d, r in got if d == "top"]
    assert tops == sorted(tops, reverse=True) and min(tops) >= 90
    decile_ids = {id(r) for d, r in got if d == "decile"}
    assert not decile_ids & {id(r) for d, r in got if d == "top"}  # no site is tested twice


def test_ranks_and_spearman():
    assert ts.ranks([3.0, 1.0, 1.0, 2.0]) == [4.0, 1.5, 1.5, 3.0]
    assert ts.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert ts.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert ts.spearman([1, 1, 1], [1, 2, 3]) is None


def _site(kind, layer, index, traffic, weight_norm=0.5):
    return {"table": "site", "kind": kind, "layer": layer, "index": index, "traffic": traffic,
            "unweighted": traffic, "count": 1, "loss_fisher": 0.1,
            "weight_norm": weight_norm}  # fmt: skip


def test_tested_sites_join_the_map_and_the_measures():
    rows = [_site("mlp", 2, 7, 3.0, 0.25),
            {"table": "test", "kind": "mlp", "site_layer": 2, "index": 7, "flips": 4,
             "mean_drift": 0.01}]  # fmt: skip
    (got,) = ts.tested_sites(rows)
    assert got["traffic"] == 3.0 and got["inverse_weight_norm"] == 4.0 and got["flips"] == 4
    assert got["draw"] is None  # a row from before the draws were named


def test_correlation_of_a_perfect_predictor():
    sites = [{"traffic": float(i), "flips": i * 2} for i in range(12)]
    got = ts.correlation(sites, "traffic", "flips")
    assert got["rho"] == 1.0 and got["shuffle_p"] < 0.01 and got["ci95"][0] == 1.0


def test_known_cases_rank_the_copy_heads_and_find_the_dormant_neurons():
    rows = [_site("head", layer, h, float(10 * layer + h)) for layer in range(6) for h in range(8)]
    rows += [_site("mlp", 0, j, float(j)) for j in range(4)] + [_site("mlp", 1, 0, 0.0)]
    activity = [{"layer": 0, "index": j, "positive": 0.5 if j < 3 else 0.001,
                 "negative": 0.5 if j < 3 else 0.002} for j in range(4)]  # fmt: skip
    got = ts.known_cases(rows, activity)
    assert got["heads"] == 48 and got["copy_head_ranks"]["L4/H7"] == 9
    assert got["copy_head_ranks_in_layer"] == {"L3/H3": 5, "L4/H0": 8, "L4/H7": 1}
    assert got["neurons"] == 4  # the last layer's neuron is left out
    assert got["dormant"] == [{"layer": 0, "index": 3, "traffic": 3.0, "rank": 1}]
    assert got["median_traffic_rest"] == 1.0


def test_draw_summary_keeps_the_draws_apart():
    sites = [{"draw": "decile", "flips": 0, "traffic": 1.0, "mean_drift": 0.1},
             {"draw": "decile", "flips": 2, "traffic": 3.0, "mean_drift": 0.3},
             {"draw": "top", "flips": 5, "traffic": 9.0, "mean_drift": 0.5}]  # fmt: skip
    decile, top = ts.draw_summary(sites)
    assert decile == {"table": "draw", "draw": "decile", "sites": 2, "flips": 2,
                      "sites_with_flips": 1, "median_traffic": 2.0, "median_drift": 0.2,
                      "max_flips": 2}  # fmt: skip
    assert top["draw"] == "top" and top["max_flips"] == 5


def test_within_layer_ranks_take_depth_away():
    # The measure follows the layer only, so a predictor equal to the layer gets no credit.
    sites = [{"kind": "mlp", "layer": layer, "traffic": float(layer), "flips": layer * 10 + i}
             for layer in range(3) for i in range(5)]  # fmt: skip
    assert ts.within_layer(sites, "traffic", "flips")["rho"] is None
    assert ts.within_layer(sites, "traffic", "flips")["shuffle_p"] is None
    for s in sites:
        s["traffic"] = float(s["flips"] % 10)  # follows the order within each layer
    got = ts.within_layer(sites, "traffic", "flips")
    assert got["rho"] == pytest.approx(1.0) and got["shuffle_p"] < 0.01


def test_difference_of_a_better_predictor_is_positive():
    sites = [
        {"traffic": float(i), "loss_fisher": float((i * 7) % 12), "flips": i} for i in range(12)
    ]
    got = ts.difference(sites, "loss_fisher", "flips")
    assert got["difference"] > 0 and got["ci95"][0] > 0
    same = ts.difference(
        [{**s, "loss_fisher": s["traffic"]} for s in sites], "loss_fisher", "flips"
    )
    assert same["difference"] == 0 and same["ci95"] == [0, 0]
