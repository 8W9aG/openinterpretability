import pytest
import torch

from openinterp.core.schemas import ActivatingExample
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.input_attribution import (
    DISTANCE_BANDS,
    _pole_values,
    ablate_positions,
    input_attribution,
    integrated_gradients,
    reference,
)
from tests.conftest import FakeTokenizer

TEXT = "abcdefgh"


def example(runner, text, layer, neuron, pole):
    strengths = runner.neuron_activations(text, layer, neuron, pole)[0].tolist()
    peak = max(range(len(strengths)), key=strengths.__getitem__)
    return ActivatingExample(
        text=text,
        tokens=list(text),
        activations=strengths,
        max_activation=strengths[peak],
        max_token_index=peak,
    )


def peak_pole(runner, layer, neuron):
    """The pole of ``neuron`` that is active at the peak of ``TEXT``."""
    act = runner.neuron_activations(TEXT, layer, neuron)[0]
    return "positive" if act.max() > -act.min() else "negative"


@pytest.fixture
def gpt2_runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def test_pole_values_match_activations_and_stop_early(gpt2_runner):
    ids = gpt2_runner.encode(TEXT)["input_ids"]
    act = gpt2_runner.neuron_activations(ids, 0, 5)
    with torch.no_grad():
        pos = _pole_values(gpt2_runner, ids, 0, 5, "positive", lambda _out: None)
        neg = _pole_values(gpt2_runner, ids, 0, 5, "negative", lambda _out: None)
    torch.testing.assert_close(pos, act)
    torch.testing.assert_close(neg, -act)


def test_integrated_gradients_completeness(gpt2_runner):
    """Attributions sum to f(x) − f(baseline) once the path integral converges."""
    runner, layer, neuron = gpt2_runner, 1, 7
    pole = peak_pole(runner, layer, neuron)
    ids = runner.encode(TEXT)["input_ids"]
    baseline = runner.model.get_input_embeddings().weight[0].detach()
    with torch.no_grad():
        full = _pole_values(runner, ids, layer, neuron, pole, lambda _out: None)[0, -1]
    for keep_first in (True, False):
        with torch.no_grad():
            edit = lambda o, k=keep_first: reference(o, baseline, k)  # noqa: E731
            empty = _pole_values(runner, ids, layer, neuron, pole, edit)[0, -1]
        attr = integrated_gradients(
            runner, ids, layer, neuron, pole, baseline, steps=128, keep_first=keep_first
        )
        assert attr.shape == (len(TEXT),)
        assert float(attr.sum()) == pytest.approx(float(full - empty), rel=1e-2, abs=1e-4)
        assert attr[1:-1].abs().sum() > 0  # attention carries context into the last token
        assert (attr[0] == 0) == keep_first  # the attention sink stays put


def test_ablate_positions_matches_editing_the_input(gpt2_runner):
    runner, layer, neuron = gpt2_runner, 1, 7
    ids = runner.encode(TEXT)["input_ids"]
    baseline = runner.model.get_input_embeddings().weight[0].detach()
    ablated = ablate_positions(runner, ids, layer, neuron, "positive", baseline, [0, 3, 7])
    for value, pos in zip(ablated.tolist(), [0, 3, 7], strict=True):
        edited = ids.clone()
        edited[0, pos] = 0
        expected = runner.neuron_activations(edited, layer, neuron)[0, -1]
        assert value == pytest.approx(float(expected), abs=1e-5)


def test_mlp_only_model_attributes_everything_to_the_current_token(toy_runner):
    """Without attention, nothing but the peak token can reach the pole."""
    pole = peak_pole(toy_runner, 1, 4)
    ex = example(toy_runner, TEXT, 1, 4, pole)
    result = input_attribution(toy_runner, [ex], 1, 4, pole, steps=64)
    assert len(result.examples) == 1
    record = result.examples[0]
    assert record.tokens == ex.max_token_index + 1
    assert record.top[0].offset == 0 and record.top[0].attribution == pytest.approx(1, abs=0.02)
    assert record.top[0].ablation_effect == pytest.approx(1, abs=1e-3)
    assert all(t.attribution == 0 and t.ablation_effect == 0 for t in record.top[1:])
    assert result.current_token_share == pytest.approx(1.0)
    assert result.mean_distance == 0
    assert [(d.min_distance, d.max_distance) for d in result.distances] == list(DISTANCE_BANDS)


def test_input_attribution_on_gpt2(gpt2_runner):
    runner = gpt2_runner
    pole = peak_pole(runner, 1, 7)
    examples = [example(runner, t, 1, 7, pole) for t in (TEXT, "hgfedcba", "aabbccdd")]
    result = input_attribution(runner, examples, 1, 7, pole, n_examples=2, steps=4, top_k=3)
    assert 1 <= len(result.examples) <= 2 and result.converged == len(result.examples)
    assert all(4 <= ex.steps <= 128 and ex.completeness_error <= 0.1 for ex in result.examples)
    assert sum(d.share for d in result.distances) == pytest.approx(1.0, abs=1e-3)
    for record in result.examples:
        assert len(record.top) == min(3, record.tokens)
        assert all(-record.tokens < t.offset <= 0 for t in record.top)
        assert all(t.ablation_effect is not None for t in record.top)
        attributions = [abs(t.attribution) for t in record.top]
        assert attributions == sorted(attributions, reverse=True)
    assert result.ablation_agreement is None or -1 <= result.ablation_agreement <= 1
    assert result.baseline_token == FakeTokenizer().decode([0])


def test_examples_the_baseline_activates_are_skipped(toy_runner):
    """A pole the reference input activates as strongly has no gap to attribute."""
    toy_runner.tokenizer.pad_token_id = toy_runner.tokenizer.encode("a")[0]
    ex = example(toy_runner, "aaa", 1, 4, peak_pole(toy_runner, 1, 4))
    result = input_attribution(toy_runner, [ex], 1, 4, peak_pole(toy_runner, 1, 4))
    assert result.examples == [] and result.distances == [] and result.ablation_agreement is None
    assert result.current_token_share is None and result.mean_distance is None


def test_unconverged_examples_are_kept_but_not_aggregated(gpt2_runner):
    runner = gpt2_runner
    pole = peak_pole(runner, 1, 7)
    examples = [example(runner, t, 1, 7, pole) for t in (TEXT, "hgfedcba", "aabbccdd")]
    result = input_attribution(runner, examples, 1, 7, pole, steps=1, max_steps=1, tolerance=1e-9)
    assert result.examples
    assert all(not r.converged and r.steps == 1 and r.top for r in result.examples)
    assert result.converged == 0 and result.distances == [] and result.ablation_agreement is None
