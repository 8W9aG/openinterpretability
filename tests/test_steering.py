import pytest
import torch

from openinterp.core.schemas import HeadIdentifier, NeuronIdentifier
from openinterp.core.steering import SteeringSample, SteeringSamples
from openinterp.harness.hooks import NeuronHookRunner, isolate_pole
from openinterp.harness.steering import steering_interventions, steering_samples
from openinterp.harness.units import HeadUnit, NeuronUnit
from tests.conftest import N_LAYERS, FakeTokenizer

PROMPTS = ("abc", "hello there")


def unit(pole, layer=1, idx=4):
    return NeuronUnit(NeuronIdentifier(model_name="toy", layer=layer, neuron_idx=idx, pole=pole))


def test_samples_cover_every_prompt_and_multiple(toy_runner):
    samples = steering_samples(toy_runner, unit("positive"), 1.5, prompts=PROMPTS)
    assert [s.prompt for s in samples.samples] == list(PROMPTS)
    assert samples.multiples == [1.0, 2.0, 4.0] and samples.clamp_unit == 1.5
    assert all(len(s.clamped) == 3 and s.baseline for s in samples.samples)
    assert steering_samples(toy_runner, unit("positive"), 1.5, prompts=PROMPTS) == samples


def greedy(runner, ids, steps):
    seq = torch.tensor([ids])
    for _ in range(steps):
        seq = torch.cat([seq, runner.forward({"input_ids": seq})[:, -1:].argmax(-1)], dim=1)
    return runner.tokenizer.decode(seq[0, len(ids) :].tolist())


def test_the_baseline_row_is_the_unpatched_greedy_continuation(toy_runner):
    ids = toy_runner.tokenizer.encode("abc")
    samples = steering_samples(
        toy_runner, unit("negative", layer=N_LAYERS - 1), 2.0, prompts=("abc",), max_new_tokens=5
    )
    assert samples.samples[0].baseline == greedy(toy_runner, ids, 5)


def test_generate_decodes_left_padded_prompts_as_if_alone(tiny_gpt2):
    """A model with ``generate`` decodes every prompt in one left-padded batch; each
    baseline must match that prompt's own unpadded greedy continuation."""
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    ident = NeuronIdentifier(model_name="tiny", layer=1, neuron_idx=3, pole="positive")
    samples = steering_samples(runner, NeuronUnit(ident), 1.0, prompts=PROMPTS, max_new_tokens=6)
    for s in samples.samples:
        assert s.baseline == greedy(runner, runner.tokenizer.encode(s.prompt), 6)
    assert any(c != s.baseline for s in samples.samples for c in s.clamped)


@pytest.mark.parametrize("pole", ["positive", "negative"])
def test_rows_are_baseline_ablated_then_clamped_at_every_position(toy_runner, pole):
    ablate = torch.tensor([False, True, False, False])
    clamp = torch.tensor([float("nan"), float("nan"), 1.0, 3.0])
    rows = toy_runner.encode(["hello there"] * 4)
    _, clean = toy_runner.run_with_cache(rows, [1])
    with toy_runner.hooks(**steering_interventions(toy_runner, unit(pole), ablate, clamp)):
        _, steered = toy_runner.run_with_cache(rows, [1])  # recorded after the steering hook
    clean, steered = clean[1], steered[1]
    torch.testing.assert_close(steered[0], clean[0])
    expected = clean[1].clone()
    expected[:, 4] -= isolate_pole(clean[1, :, 4], pole)
    torch.testing.assert_close(steered[1], expected)
    sign = 1.0 if pole == "positive" else -1.0
    for row, value in zip((2, 3), clamp[2:].tolist(), strict=True):
        assert torch.all(steered[row, :, 4] == sign * value)
        others = torch.arange(steered.shape[-1]) != 4
        torch.testing.assert_close(steered[row][:, others], clean[row][:, others])


def test_heads_have_no_clamped_setting(toy_runner):
    head = HeadUnit(HeadIdentifier(model_name="toy", layer=1, head_idx=0), (0.0,))
    none = steering_interventions(toy_runner, head, torch.tensor([False]), torch.tensor([1.0]))
    assert none is None
    with pytest.raises(ValueError, match="clamped-high"):
        steering_samples(toy_runner, head, 1.0, prompts=PROMPTS)


def test_one_clamped_text_per_multiple():
    with pytest.raises(ValueError, match="2 clamped texts for 3 multiples"):
        SteeringSamples(
            clamp_unit=1.0,
            multiples=[1.0, 2.0, 4.0],
            max_new_tokens=4,
            samples=[SteeringSample(prompt="p", baseline="a", ablated="b", clamped=["c", "d"])],
        )


def test_samples_rendered_on_both_pages(tmp_path, record):
    from openinterp.builder.compiler import compile_site
    from openinterp.builder.wiki import write_experiment_page

    steering = SteeringSamples(
        clamp_unit=2.5,
        multiples=[1.0, 4.0],
        max_new_tokens=24,
        samples=[
            SteeringSample(
                prompt="The weather today is",
                baseline=" sunny.",
                ablated=" a|b\nc",
                clamped=[" sunny.", " quoted “speech”"],
            )
        ],
    )
    sub = record.submission.model_copy(update={"steering": steering})
    rec = record.model_copy(update={"submission": sub})
    md = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "### Steering" in md
    assert "| Prompt | Ablated | Baseline | Clamped ×1 | Clamped ×4 |" in md
    assert "| The weather today is |  a\\|b ⏎ c |  sunny. |  sunny. |  quoted “speech” |" in md
    assert "### Steering" not in write_experiment_page(record, tmp_path / "old").read_text()

    compile_site([rec], tmp_path / "site")
    (page,) = (tmp_path / "site" / "models").rglob("N117.html")
    html = page.read_text()
    assert "Steering" in html and "quoted “speech”" in html and "Clamped ×4" in html
    assert 'title="Same as the baseline"> sunny.</p>' in html  # ×1 changed nothing
    assert "lg:grid-cols-4" in html
