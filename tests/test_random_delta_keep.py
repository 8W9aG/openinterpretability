"""The helpers of scripts/random_delta_keep.py (#5154)."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import random_delta_keep as rdk  # noqa: E402


class Block(nn.Module):
    """A block with a Qwen2 MLP's names, and one other weight."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(3, 4, bias=False)
        self.mlp.up_proj = nn.Linear(3, 4, bias=False)
        self.mlp.down_proj = nn.Linear(4, 3, bias=False)
        self.attn = nn.Linear(3, 3)


def _pair():
    torch.manual_seed(0)
    base, other = Block(), Block()
    return base, other


def test_deltas_are_the_other_weights_minus_the_base():
    base, other = _pair()
    delta = rdk.deltas(base, other)
    assert set(delta) == {n for n, _ in base.named_parameters()}
    for name, weight in base.named_parameters():
        assert torch.equal(delta[name], dict(other.named_parameters())[name] - weight)


def test_mlp_names_are_the_three_matrices():
    assert rdk.is_mlp("model.layers.3.mlp.down_proj.weight")
    assert not rdk.is_mlp("model.layers.3.self_attn.q_proj.weight")
    assert not rdk.is_mlp("model.layers.3.mlp.down_proj.bias")


@pytest.mark.parametrize(
    ("condition", "names"),
    [
        ("instruct", {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                      "attn.weight", "attn.bias"}),
        ("instruct_mlp", {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}),
        ("random_mlp", {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}),
        ("random_all", {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                        "attn.weight", "attn.bias"}),
    ],
)  # fmt: skip
def test_each_condition_changes_its_own_weights(condition, names):
    delta = rdk.deltas(*_pair())
    change = rdk.change_for(condition, delta, seed=1)
    assert set(change) == names
    for name, tensor in change.items():
        assert tensor.shape == delta[name].shape
        assert float(tensor.norm()) == pytest.approx(float(delta[name].norm()), rel=1e-5)
        if condition.startswith("instruct"):
            assert torch.equal(tensor, delta[name])
        else:
            assert not torch.equal(tensor, delta[name])


def test_random_changes_follow_their_seed():
    delta = rdk.deltas(*_pair())
    one, again = rdk.change_for("random_mlp", delta, 3), rdk.change_for("random_mlp", delta, 3)
    other = rdk.change_for("random_mlp", delta, 4)
    name = "mlp.up_proj.weight"
    assert torch.equal(one[name], again[name]) and not torch.equal(one[name], other[name])


def test_an_unknown_condition_is_an_error():
    with pytest.raises(ValueError, match="unknown condition"):
        rdk.change_for("instruct_attention", {}, 0)


def test_the_weights_come_back_exactly_after_a_change():
    base, other = _pair()
    before = {n: w.detach().clone() for n, w in base.named_parameters()}
    change = rdk.change_for("instruct", rdk.deltas(base, other))
    with rdk.changed(base, change) as model:
        for name, weight in model.named_parameters():
            assert torch.allclose(weight, dict(other.named_parameters())[name], atol=1e-6)
    for name, weight in base.named_parameters():
        assert torch.equal(weight, before[name])


def test_the_weights_come_back_after_an_error():
    base, other = _pair()
    before = {n: w.detach().clone() for n, w in base.named_parameters()}
    with (
        pytest.raises(RuntimeError),
        rdk.changed(base, rdk.change_for("random_all", rdk.deltas(base, other))),
    ):
        raise RuntimeError("a failed condition")
    for name, weight in base.named_parameters():
        assert torch.equal(weight, before[name])


def test_summary_counts_only_the_poles_that_reproduce():
    rows = [
        {"slug": "a", "base_passes": True, "keeps": True},
        {"slug": "b", "base_passes": True, "keeps": False},
        {"slug": "c", "base_passes": False, "keeps": False},
    ]
    out = rdk.summary(rows, condition="instruct")
    assert out["poles"] == 3 and out["reproduced_on_base"] == 2 and out["keeps"] == 1
    assert out["keep_share"] == 0.5 and out["flipped"] == ["b"] and out["condition"] == "instruct"
    assert rdk.summary([], condition="x")["keep_share"] is None


class _Ident:
    def __init__(self, layer, neuron, pole):
        self.layer, self.neuron_idx, self.pole = layer, neuron, pole
        self.slug = f"L{layer}-N{neuron}-{pole}"


class _Record:
    def __init__(self, layer, neuron, pole):
        self.submission = type("S", (), {"identifier": _Ident(layer, neuron, pole)})()


def test_run_condition_reads_each_pole_on_the_changed_model(monkeypatch):
    base_model, other = _pair()
    runner = type("R", (), {"model": base_model})()
    records = [_Record(0, 1, "positive"), _Record(0, 2, "negative"), _Record(1, 0, "positive")]
    columns = {0: torch.tensor([[1.0, -2.0], [3.0, 0.5]]), 1: torch.tensor([[0.0], [-1.0]])}
    changed_by = {"on": False}

    def fake_columns(_runner, _batches, layer, neurons):
        # The changed model doubles every activation, so each ratio is 2 where a pole has a peak.
        return columns[layer][:, : len(neurons)] * (2.0 if changed_by["on"] else 1.0)

    monkeypatch.setattr(rdk, "layer_columns", fake_columns)
    base = rdk.base_columns(runner, [], records)
    assert base[0][0] == [1, 2] and base[1][0] == [0]
    changed_by["on"] = True
    monkeypatch.setattr(rdk, "probe_activations", lambda _r, _recs: {})
    monkeypatch.setattr(rdk, "first_position_activations", lambda _r, _recs: {})
    monkeypatch.setattr(
        rdk, "cached_readings", lambda _r, rec, _c, _f, scale: {"passes": scale == 2.0}
    )
    on_base = {r.submission.identifier.slug: {"passes": True} for r in records}
    change = rdk.change_for("instruct", rdk.deltas(base_model, other))
    summary, *rows = rdk.run_condition(
        runner, [], records, base, on_base, change, {"condition": "x"}
    )
    by_slug = {r["slug"]: r for r in rows}
    assert by_slug["L0-N1-positive"]["ratio"] == 2.0 and by_slug["L0-N1-positive"]["keeps"]
    assert by_slug["L0-N2-negative"]["ratio"] == 2.0
    # Neuron 0 of layer 1 has no positive peak on the base, so it has no ratio and scale 1.
    assert by_slug["L1-N0-positive"]["ratio"] is None and not by_slug["L1-N0-positive"]["keeps"]
    assert summary["keeps"] == 2 and summary["flipped"] == ["L1-N0-positive"]
    norm = float(torch.sqrt(sum((t.double() ** 2).sum() for t in change.values())))
    assert summary["change_norm"] == pytest.approx(norm, abs=1e-4)


class _Result:
    def __init__(self, prompt, kind="hard_negative"):
        self.prompt, self.kind = prompt, kind


class _ProbedRecord(_Record):
    def __init__(self, layer, neuron, prompts):
        super().__init__(layer, neuron, "positive")
        results = [_Result(p) for p in prompts] + [_Result("a counterfactual", "counterfactual")]
        self.verdict = type("V", (), {"adversarial_test_results": results})()


class _Submission:
    def __init__(self, identifier, first_token_floor=None, max_activation=400.0):
        self.identifier, self.first_token_floor = identifier, first_token_floor
        self.max_activation, self.counterfactuals = max_activation, []
        self.head_profile = self.dictionary_profile = None

    def model_copy(self, update):
        out = _Submission(self.identifier, self.first_token_floor, self.max_activation)
        vars(out).update(update)
        return out


def _inputs(runner, inputs):
    return inputs if isinstance(inputs, dict) else runner.encode(inputs)


class _CausalRunner:
    """Each token's activation at (layer, neuron) depends only on the tokens up to it."""

    def __init__(self, vocab=40):
        self.passes = []
        self.model = nn.Linear(1, 1)  # on the CPU, so map_layer_major runs in order
        self.device = torch.device("cpu")
        self.tokenizer = type("T", (), {"__len__": lambda _: vocab, "all_special_ids": [0]})()

    def encode(self, text):
        text = text if isinstance(text, str) else text[0]  # prompt_tokens passes [text]
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def layer_activations(self, inputs, layers):
        ids = inputs["input_ids"].float()
        self.passes.append(tuple(ids.shape))
        running = ids.cumsum(dim=1)
        return {
            layer: torch.stack([running * (layer + 1) + n for n in range(4)], dim=-1)
            for layer in layers
        }


def test_probe_activations_match_one_text_at_a_time():
    runner = _CausalRunner()
    records = [
        _ProbedRecord(0, 1, ["ab", "a much longer text"]),
        _ProbedRecord(2, 3, ["xyz", "ab"]),
    ]
    cache = rdk.probe_activations(runner, records, max_tokens=8)
    assert set(cache) == {("ab", 0, 1), ("a much longer text", 0, 1), ("xyz", 2, 3), ("ab", 2, 3)}
    for (text, layer, neuron), acts in cache.items():
        alone = _CausalRunner().layer_activations(runner.encode(text), [layer])[layer][0, :, neuron]
        assert torch.equal(acts, alone)
    # "ab" and "xyz" share a padded pass of 2 x 3 tokens; the long text runs alone.
    assert runner.passes == [(2, 3), (1, 18)]


def test_a_cached_unit_reads_its_pole_from_the_cache():
    from openinterpretability.core.schemas import NeuronIdentifier
    from openinterpretability.harness.units import NeuronUnit

    ident = NeuronIdentifier(model_name="m", layer=1, neuron_idx=2, pole="negative")
    cache = {("t", 1, 2): torch.tensor([1.0, -3.0])}
    unit = rdk.CachedUnit(NeuronUnit(ident, skip_first=True), cache)
    assert unit.skip_first
    assert torch.equal(unit.strengths(None, "t"), torch.tensor([[0.0, 3.0]]))


def test_cached_readings_match_readings(monkeypatch):
    from openinterpretability.agents import review_measurements
    from openinterpretability.core.schemas import NeuronIdentifier
    from openinterpretability.harness.hooks import pole_strength
    from openinterpretability.harness.units import NeuronUnit
    from openinterpretability.interference.keep import readings

    ident = NeuronIdentifier(model_name="m", layer=0, neuron_idx=1, pole="positive")
    runner = _CausalRunner()
    runner.neuron_activations = lambda text, layer, neuron, pole: pole_strength(
        runner.layer_activations(_inputs(runner, text), [layer])[layer][..., neuron], pole
    )
    record = _ProbedRecord(0, 1, [])
    record.submission = _Submission(ident, first_token_floor=False)
    texts = [("a", False), ("ab", False), ("zz", True), ("abc", True), ("~~~", True)]
    record.verdict.policy = None
    record.verdict.adversarial_test_results = [
        type("R", (), {"prompt": t, "expected_active": e,
                       "kind": "positive_holdout" if e else "hard_negative"})()
        for t, e in texts
    ]  # fmt: skip
    monkeypatch.setattr(
        review_measurements, "settle_first_token_floor",
        lambda sub, _runner, _policy: (sub, NeuronUnit(ident), 400.0),
    )  # fmt: skip
    cache = rdk.probe_activations(runner, [record])
    for factor in (0.5, 1.0, 2.0):
        assert rdk.cached_readings(runner, record, cache, {}, factor) == readings(
            runner, record, factor
        )


def _floor_runner(vocab=40):
    from openinterpretability.harness.hooks import pole_strength

    runner = _CausalRunner(vocab)

    def neuron_activations(inputs, layer, neuron, pole=None):
        act = runner.layer_activations(_inputs(runner, inputs), [layer])[layer][..., neuron]
        return act if pole is None else pole_strength(act, pole)

    runner.neuron_activations = neuron_activations
    return runner


def test_one_vocabulary_sweep_matches_each_neuron_alone():
    from openinterpretability.harness.first_position import strengths_of

    runner = _floor_runner()
    records = [_Record(0, 1, "positive"), _Record(2, 3, "negative"), _Record(2, 0, "positive")]
    firsts = rdk.first_position_activations(runner, records, batch_size=16)
    assert set(firsts) == {(0, 1), (2, 3), (2, 0)}
    assert len(runner.passes) == 3  # 39 tokens in passes of 16, each pass for both layers
    for (layer, neuron), raw in firsts.items():
        alone = strengths_of(
            runner, lambda x, at=layer, n=neuron: runner.neuron_activations(x, at, n)
        )
        assert torch.equal(raw, alone)


@pytest.mark.parametrize("scale", [10.0, 60.0, 120.0])
def test_the_floor_from_the_sweep_is_the_floor_of_settle(scale):
    from openinterpretability.agents.review_measurements import settle_first_token_floor
    from openinterpretability.core.schemas import NeuronIdentifier, ReviewPolicy

    runner, policy = _floor_runner(), ReviewPolicy()
    ident = NeuronIdentifier(model_name="m", layer=0, neuron_idx=1, pole="positive")
    record = _Record(0, 1, "positive")
    record.submission = _Submission(ident, max_activation=scale)
    firsts = rdk.first_position_activations(runner, [record])
    unit, got_scale = rdk.settle_cached(record, runner, policy, firsts)
    _, expected, expected_scale = settle_first_token_floor(record.submission, runner, policy)
    assert (unit, got_scale) == (expected, expected_scale)
    assert unit.skip_first == (scale < 100)  # the line is a quarter of the scale


def test_a_pass_fills_its_memory_budget(monkeypatch):
    monkeypatch.setattr(rdk, "PASS_BYTES", 4 * 10 * 2 * 100)
    records = [_Record(0, 1, "positive"), _Record(3, 2, "positive"), _Record(3, 5, "negative")]
    runner = type("R", (), {"model": type("M", (), {"config": type("C", (), {})()})()})()
    assert rdk.pass_tokens(runner, records) == rdk.PASS_TOKENS  # no intermediate_size
    runner.model.config.intermediate_size = 10
    assert rdk.pass_tokens(runner, records) == 100  # 2 layers of 10 float32 values a token
    runner.model.config.intermediate_size = 10**6
    assert rdk.pass_tokens(runner, records) == 1


def test_base_rows_give_each_pole_its_base_readings():
    records = [_Record(0, 1, "positive"), _Record(4, 2, "negative")]
    on_base = {"L0-N1-positive": {"passes": True}, "L4-N2-negative": {"passes": False}}
    rows = rdk.base_rows(records, on_base, "m")
    assert [r["table"] for r in rows] == ["base", "base"]
    assert rows[1] == {"table": "base", "model": "m", "slug": "L4-N2-negative", "layer": 4,
                       "pole": "negative", "readings": {"passes": False}}  # fmt: skip
