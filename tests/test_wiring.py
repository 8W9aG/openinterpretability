"""Wiring from weights and from activations, on toy, GPT-2 and Qwen2 models (no download)."""

import pytest
import torch

from openinterp.core.schemas import FeatureIdentifier, HeadIdentifier, NeuronIdentifier
from openinterp.harness.features import scan_feature
from openinterp.harness.hooks import NeuronHookRunner, ablate_pole, pole_strength
from openinterp.harness.units import FeatureUnit, HeadUnit, NeuronUnit
from openinterp.harness.wiring import attention_reads, target_of, weight_wiring
from openinterp.harness.wiring_activations import _downstream_head_effects, activation_wiring
from tests.conftest import N_LAYERS, FakeTokenizer
from tests.test_feature_audit import TEXTS, spec  # noqa: F401  (fixture)

PROMPTS = ["hello world", "abcabc", "the cat sat"]


def neuron(layer: int, idx: int, pole: str = "positive") -> NeuronUnit:
    return NeuronUnit(NeuronIdentifier(model_name="toy", layer=layer, neuron_idx=idx, pole=pole))


def head(layer: int, idx: int) -> HeadUnit:
    return HeadUnit(HeadIdentifier(model_name="toy", layer=layer, head_idx=idx), (0.0,) * 16)


@pytest.fixture
def gpt2_runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


@pytest.fixture
def qwen_runner():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return NeuronHookRunner(Qwen2ForCausalLM(config).eval(), FakeTokenizer())


def test_weights_match_virtual_weights_by_hand(gpt2_runner, tiny_gpt2):
    block0, block1 = tiny_gpt2.transformer.h
    wiring = weight_wiring(gpt2_runner, neuron(1, 5), k=3)
    read = block1.mlp.c_fc.weight[:, 5].detach() * block1.ln_2.weight.detach()
    read = read - read.mean()
    expected = block0.mlp.c_proj.weight.detach() @ (read / read.norm())  # every L0 neuron
    top = wiring.upstream_neurons[0]
    assert (top.kind, top.layer, top.channels.keys()) == ("mlp_neuron", 0, {"in"})
    assert top.index == int(expected.abs().argmax())
    assert top.weight == pytest.approx(float(expected[top.index]), abs=1e-5)
    assert len(wiring.upstream_heads) == 3 and all(e.weight >= 0 for e in wiring.upstream_heads)
    assert {e.layer for e in wiring.upstream_heads} <= {0, 1}
    assert wiring.downstream_neurons == [] and wiring.downstream_heads == []


def test_negative_pole_flips_downstream_signs_only(gpt2_runner):
    pos = weight_wiring(gpt2_runner, neuron(0, 3, "positive"))
    neg = weight_wiring(gpt2_runner, neuron(0, 3, "negative"))
    assert [e.weight for e in neg.downstream_neurons] == [-e.weight for e in pos.downstream_neurons]
    assert neg.downstream_heads == pos.downstream_heads and neg.upstream_heads == pos.upstream_heads
    assert neg.upstream_neurons == pos.upstream_neurons  # signed by the read, not the pole


def test_gated_mlp_and_grouped_query_attention(qwen_runner):
    reads = attention_reads(qwen_runner, 1)
    assert reads["q"].shape == (4, 8, 32) and reads["k"].shape == (4, 8, 32)
    assert torch.equal(reads["k"][0], reads["k"][1])
    assert not torch.equal(reads["q"][0], reads["q"][1])
    wiring = weight_wiring(qwen_runner, neuron(1, 7))
    edge = wiring.upstream_neurons[0]
    assert edge.channels.keys() == {"gate", "up"}
    assert edge.weight == max(edge.channels.values(), key=abs)
    assert {e.channels.keys() == {"q", "k", "v"} for e in wiring.downstream_heads} == {True}
    assert len({e.index for e in wiring.downstream_heads if e.layer == 2}) == 4

    acts = activation_wiring(qwen_runner, neuron(1, 7), PROMPTS, threshold=0.0)
    assert acts.prompts == 3 and {e.layer for e in acts.downstream_heads} == {2}


def test_activation_downstream_is_the_direct_effect_of_ablating_the_pole(toy_runner):
    # The toy model has no attention, so a later layer sees the pole only directly.
    unit = neuron(0, 2, "negative")
    wiring = activation_wiring(toy_runner, unit, ["hello"], threshold=0.0, k=50)
    assert wiring.prompts == 1 and wiring.upstream_heads is None
    _, clean = toy_runner.run_with_cache("hello", [0, 1])
    t = int(pole_strength(clean[0][0, :, 2], "negative").argmax())
    with toy_runner.hooks({0: ablate_pole(2, "negative")}):
        _, ablated = toy_runner.run_with_cache("hello", [1])
    effect = clean[1][0, t] - ablated[1][0, t]
    for e in (e for e in wiring.downstream_neurons if e.layer == 1):
        assert e.weight == pytest.approx(float(effect[e.index]), abs=1e-5)


def scaled(runner, layer, idx, scale):
    def shrink(act):
        act = act.clone()
        act[..., idx] *= scale
        return act

    return runner.hooks({layer: shrink})


def test_activation_upstream_is_first_order_in_each_write(toy_runner):
    unit = neuron(N_LAYERS - 1, 4)
    wiring = activation_wiring(toy_runner, unit, ["hello"], threshold=0.0, k=100)
    edge = next(e for e in wiring.upstream_neurons if e.layer == N_LAYERS - 2)

    def strength(scale: float) -> float:
        # Scaling the neuron at every position changes later layers only through this one.
        with scaled(toy_runner, edge.layer, edge.index, scale):
            return toy_runner.max_pole_strength("hello", unit.ident.layer, 4, "positive")

    eps = 1e-3
    slope = (strength(1.0) - strength(1 - eps)) / eps
    assert slope == pytest.approx(edge.weight, rel=1e-2, abs=1e-5)


def test_activation_wiring_batches_and_skips_inactive_prompts(gpt2_runner):
    one_by_one = [activation_wiring(gpt2_runner, neuron(1, 5), [p], 0.0, k=200) for p in PROMPTS]
    wiring = activation_wiring(gpt2_runner, neuron(1, 5), PROMPTS, threshold=0.0, k=200)
    assert wiring.prompts == len(PROMPTS) and wiring.downstream_heads == []
    assert len(wiring.upstream_heads) == 4 and len(wiring.upstream_neurons) == 128
    by_unit = {(e.layer, e.index): e.weight for e in wiring.upstream_neurons}
    for e in wiring.upstream_neurons[:5]:  # the batch mean is the mean of single prompts
        alone = [{(x.layer, x.index): x.weight for x in w.upstream_neurons} for w in one_by_one]
        assert by_unit[(e.layer, e.index)] == pytest.approx(
            sum(a[(e.layer, e.index)] for a in alone) / 3, abs=1e-5
        )
    assert activation_wiring(gpt2_runner, neuron(1, 5), PROMPTS, threshold=1e9) is None


def test_downstream_head_value_channel_matches_attention_by_hand(gpt2_runner, tiny_gpt2):
    """Removing the write from v alone leaves the pattern, so Δz_t = Σ_s A_ts W_V Δln_1(x_s)."""
    tiny_gpt2.config._attn_implementation = "eager"
    unit = neuron(0, 3)
    t = target_of(gpt2_runner, unit)
    encoded = gpt2_runner.encode(["hello world"])
    from openinterp.harness.wiring_activations import _clean_pass

    cap = _clean_pass(gpt2_runner, t, encoded, attention=True)
    delta = t.written(cap)
    peak = int(t.strength(cap["read"]).argmax())
    effects = _downstream_head_effects(gpt2_runner, t, encoded, cap, delta, 1, torch.tensor([peak]))
    block = tiny_gpt2.transformer.h[1]
    with torch.no_grad():
        out = tiny_gpt2(**encoded, output_attentions=True)
        pattern = out.attentions[1][0, :, peak]  # [heads, seq]
        x = cap["attn_in"][1][0]
        w_v = block.attn.c_attn.weight[:, 64:]  # Conv1D: [d, 3d], v is the last third
        dv = (block.ln_1(x) - block.ln_1(x - delta[0])) @ w_v  # [seq, d]
        w_o = block.attn.c_proj.weight  # [n·hd, d]
        for h in range(2):
            dz = pattern[h] @ dv[:, h * 16 : (h + 1) * 16]
            moved = dz @ w_o[h * 16 : (h + 1) * 16]
            assert float(effects[0, h, 2]) == pytest.approx(float(moved.norm()), rel=1e-4)
    assert (effects[0, :, :2] > 0).all()  # q and k carry the write too


def test_head_wiring(gpt2_runner, tiny_gpt2):
    unit = head(1, 0)
    wiring = weight_wiring(gpt2_runner, unit, k=5)
    assert {e.layer for e in wiring.upstream_neurons} == {0}  # MLP 1 runs after the head
    assert {e.layer for e in wiring.upstream_heads} == {0}
    assert {e.layer for e in wiring.downstream_neurons} == {1}  # ...and reads its write
    assert wiring.downstream_heads == []
    assert wiring.upstream_neurons[0].channels.keys() == {"q", "k", "v"}
    assert all(e.weight >= 0 for e in wiring.upstream_neurons)

    acts = activation_wiring(gpt2_runner, unit, PROMPTS, threshold=0.0, k=200)
    assert acts.prompts == 3 and {e.layer for e in acts.downstream_neurons} == {1}
    edge = acts.upstream_neurons[0]

    def strength(scale: float) -> float:
        with scaled(gpt2_runner, 0, edge.index, scale):
            return sum(float(unit.strengths(gpt2_runner, p).max()) for p in PROMPTS) / 3

    eps = 1e-3
    slope = (strength(1.0) - strength(1 - eps)) / eps
    assert slope == pytest.approx(edge.weight, rel=2e-2, abs=1e-5)


def test_feature_wiring(gpt2_runner, spec):  # noqa: F811
    ident = FeatureIdentifier(model_name="tiny-gpt2", layer=1, dictionary="tiny-tc", feature_idx=5)
    profile = scan_feature(gpt2_runner, TEXTS, ident, spec, k=4).profile
    unit = FeatureUnit(ident, profile)
    wiring = weight_wiring(gpt2_runner, unit, k=5)
    assert {e.layer for e in wiring.upstream_neurons} == {0}
    assert {e.layer for e in wiring.upstream_heads} <= {0, 1}
    assert wiring.downstream_neurons == [] and wiring.downstream_heads == []

    acts = activation_wiring(gpt2_runner, unit, TEXTS, threshold=1e-6, k=200)
    edge = next(e for e in acts.upstream_neurons if e.layer == 0)

    def strength(scale: float) -> float:
        with scaled(gpt2_runner, 0, edge.index, scale):
            values = [unit.strengths(gpt2_runner, p).max() for p in TEXTS]
        return sum(float(v) for v in values if v >= 1e-6)

    eps = 1e-3
    slope = (strength(1.0) - strength(1 - eps)) / eps / acts.prompts
    assert slope == pytest.approx(edge.weight, rel=2e-2, abs=1e-5)


@pytest.mark.parametrize("site", ["resid_mid", "mlp_in", "mlp_in_normalized"])
def test_a_features_read_matches_its_strength_at_each_site(gpt2_runner, spec, site):  # noqa: F811
    from dataclasses import replace

    ident = FeatureIdentifier(model_name="tiny-gpt2", layer=1, dictionary="tiny-tc", feature_idx=5)
    profile = scan_feature(gpt2_runner, TEXTS, ident, replace(spec, input_site=site), k=2).profile
    unit = FeatureUnit(ident, profile)
    seen = []  # the residual stream the MLP reads: its norm's input
    ln_2 = gpt2_runner.model.transformer.h[1].ln_2
    handle = ln_2.register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
    expected = unit.strengths(gpt2_runner, TEXTS[0])
    handle.remove()
    t = target_of(gpt2_runner, unit)
    torch.testing.assert_close(t.strength(seen[-1]), expected, atol=1e-5, rtol=0)
