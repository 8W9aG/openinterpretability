"""Streaming mode: the decision to stream, and that a streamed model measures the same."""

import pytest
import torch

from openinterp.harness import memory
from openinterp.harness.memory import MemoryPlan, plan, stream_batch_tokens, stream_device_map

GB = 2**30


@pytest.mark.parametrize(
    ("mode", "need", "have", "stream"),
    [
        ("auto", 4 * GB, 8 * GB, False),  # fits: never stream by default
        ("auto", 12 * GB, 8 * GB, True),  # doesn't fit
        ("auto", None, 8 * GB, False),  # unknown need: don't guess
        ("auto", 12 * GB, None, False),  # unknown free memory: don't guess
        ("normal", 12 * GB, 8 * GB, False),
        ("stream", 4 * GB, 8 * GB, True),
    ],
)
def test_plan_streams_only_when_asked_or_needed(mode, need, have, stream):
    decision = plan(mode, need, have)
    assert isinstance(decision, MemoryPlan) and decision.stream is stream


def test_device_map_puts_each_layer_on_disk_and_the_rest_on_cpu(tiny_gpt2):
    device_map = stream_device_map(tiny_gpt2, "transformer.h")
    layers = {k for k, v in device_map.items() if v == "disk"}
    assert layers == {f"transformer.h.{i}" for i in range(len(tiny_gpt2.transformer.h))}
    assert {"transformer.wte", "transformer.ln_f", "lm_head"} <= set(device_map)
    assert all(v == "cpu" for k, v in device_map.items() if k not in layers)


def test_stream_batch_tokens_fit_half_the_free_memory():
    class Config:
        hidden_size, intermediate_size = 1024, 4096

    per_token = 4 * (4 * 1024 + 3 * 4096)  # float32
    assert stream_batch_tokens(Config, torch.float32, 2 * per_token * 10_000, 2048) == 10_000
    assert stream_batch_tokens(Config, torch.float32, 1, 2048) == 2048  # the floor
    big = stream_batch_tokens(Config, torch.float32, 1 << 50, 2048)
    assert big == memory.STREAM_BATCH_TOKENS_MAX
    assert stream_batch_tokens(Config, torch.float32, None, 2048) == 2048


@pytest.fixture
def saved_gpt2(tiny_gpt2, tmp_path):
    path = tmp_path / "gpt2"
    tiny_gpt2.save_pretrained(path)
    return path


def _runner(path, mode, monkeypatch, tmp_path):
    from openinterp.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    memory.request(mode)
    try:
        model = memory.load_causal_lm(str(path), "cpu", torch.float32, None).eval()
    finally:
        memory.request("auto")
    return NeuronHookRunner(model, FakeTokenizer())


def test_a_streamed_model_measures_what_a_loaded_one_does(saved_gpt2, monkeypatch, tmp_path):
    from openinterp.core.identifiers import NeuronIdentifier
    from openinterp.harness.dead import prove_dead
    from openinterp.harness.scan import scan_neuron
    from openinterp.harness.wiring import _matrix

    whole = _runner(saved_gpt2, "normal", monkeypatch, tmp_path)
    streamed = _runner(saved_gpt2, "stream", monkeypatch, tmp_path)
    assert not memory.is_streamed(whole.model) and memory.is_streamed(streamed.model)
    assert streamed.device.type == "cpu"

    texts = ["abcde", "hello world", "zzzz", "aaaa bbbb"]
    a = scan_neuron(whole, texts, 1, 7, "positive")
    b = scan_neuron(streamed, texts, 1, 7, "positive")
    assert a.values == b.values and a.examples == b.examples
    torch.testing.assert_close(
        whole.run_with_interventions(texts[:1]), streamed.run_with_interventions(texts[:1])
    )
    # The reads of weights inside a streamed layer, which are on disk between passes.
    assert torch.equal(whole.neuron_output_direction(1, 7), streamed.neuron_output_direction(1, 7))
    assert torch.equal(whole.head_output_weights(1, 0), streamed.head_output_weights(1, 0))
    assert torch.equal(_matrix(whole.down_proj(1)), _matrix(streamed.down_proj(1)))
    ident = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=7, pole="negative")
    assert prove_dead(whole, ident) == prove_dead(streamed, ident)
    assert not any((tmp_path / "cache").rglob("*.dat"))  # no copy of the weights written


def test_auto_loads_a_model_that_fits_whole(saved_gpt2, monkeypatch, tmp_path):
    runner = _runner(saved_gpt2, "auto", monkeypatch, tmp_path)
    assert not memory.is_streamed(runner.model)


def test_auto_streams_a_model_that_does_not_fit(saved_gpt2, monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "available_bytes", lambda: 1)  # 1 byte free
    runner = _runner(saved_gpt2, "auto", monkeypatch, tmp_path)
    assert memory.is_streamed(runner.model)


def test_a_memory_budget_overrides_what_is_free(monkeypatch):
    monkeypatch.setenv(memory.MEMORY_ENV, "1.5")
    assert memory.available_bytes() == int(1.5 * GB)


def test_streamed_attribution_recomputes_layers_and_matches(saved_gpt2, monkeypatch, tmp_path):
    """Integrated gradients through streamed layers: the same, without keeping their weights."""
    from openinterp.harness.input_attribution import input_attribution
    from openinterp.harness.scan import scan_neuron

    whole = _runner(saved_gpt2, "normal", monkeypatch, tmp_path)
    streamed = _runner(saved_gpt2, "stream", monkeypatch, tmp_path)
    texts = ["abcdefgh", "hgfedcba", "aabbccdd"]
    examples = scan_neuron(whole, texts, 1, 7, "positive").examples
    calls = []
    real = torch.utils.checkpoint.checkpoint

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)
    a = input_attribution(whole, examples, 1, 7, "positive", n_examples=2, steps=4)
    assert not calls  # a model that fits isn't recomputed
    b = input_attribution(streamed, examples, 1, 7, "positive", n_examples=2, steps=4)
    assert calls  # layer 0, below the unit's layer 1, was recomputed
    assert a == b
    assert streamed.model.config.use_cache == whole.model.config.use_cache  # restored
