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
    from openinterp.harness.environment import run_environment
    from openinterp.harness.scan import scan_neuron
    from openinterp.harness.wiring import _matrix

    whole = _runner(saved_gpt2, "normal", monkeypatch, tmp_path)
    streamed = _runner(saved_gpt2, "stream", monkeypatch, tmp_path)
    assert not memory.is_streamed(whole.model) and memory.is_streamed(streamed.model)
    assert streamed.device.type == "cpu"
    assert run_environment(streamed).streamed is True
    assert run_environment(whole).streamed is None  # unset, as in older records

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


def test_a_streamed_model_opens_each_checkpoint_file_once(saved_gpt2, monkeypatch, tmp_path):
    """Accelerate opens the file again for each tensor, which was most of a pass's time."""
    import safetensors

    streamed = _runner(saved_gpt2, "stream", monkeypatch, tmp_path)
    opened = []
    real = safetensors.safe_open

    def spy(path, *args, **kwargs):
        opened.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", spy)
    monkeypatch.setattr("accelerate.utils.offload.safe_open", spy)
    for _ in range(2):
        streamed.run_with_interventions(["abcde"])
    assert len(opened) == len(set(opened)) == 1  # one file, opened on the first read


def test_auto_loads_a_model_that_fits_whole(saved_gpt2, monkeypatch, tmp_path):
    runner = _runner(saved_gpt2, "auto", monkeypatch, tmp_path)
    assert not memory.is_streamed(runner.model)


def test_auto_streams_a_model_that_does_not_fit(saved_gpt2, monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "available_bytes", lambda device=None: 1)  # 1 byte free
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


def test_a_mixture_of_experts_model_streams_whole_layers(monkeypatch, tmp_path):
    """Its experts are fused into one tensor per layer, so a layer is read whole."""
    from transformers import MixtralConfig, MixtralForCausalLM

    config = MixtralConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=64,
    )  # fmt: skip
    torch.manual_seed(0)
    whole = MixtralForCausalLM(config).eval()
    whole.save_pretrained(tmp_path / "moe")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    memory.request("stream")
    try:
        streamed = memory.load_causal_lm(str(tmp_path / "moe"), "cpu", torch.float32, None)
    finally:
        memory.request("auto")
    assert memory.is_streamed(streamed)
    ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        torch.testing.assert_close(whole(ids).logits, streamed.eval()(ids).logits)


def test_streamed_passes_run_more_rows_and_measure_the_same(saved_gpt2, monkeypatch, tmp_path):
    """Patching, the joint ablation and the lens batch more rows when streamed.

    Rows are independent, so only float rounding changes, and there are fewer passes.
    """
    from openinterp.harness.covariance import joint_ablation
    from openinterp.harness.hooks import NeuronHookRunner
    from openinterp.harness.jlens import transported_direction
    from openinterp.harness.patcher import ActivationPatcher, metric_change

    whole = _runner(saved_gpt2, "normal", monkeypatch, tmp_path)
    streamed = _runner(saved_gpt2, "stream", monkeypatch, tmp_path)
    passes = {"n": 0}
    real = NeuronHookRunner.forward

    def counted(self, encoded):
        passes["n"] += 1
        return real(self, encoded)

    monkeypatch.setattr(NeuronHookRunner, "forward", counted)
    prompts = ["hello world", "abcdefgh", "zzzz yyyy", "aaaa bbbb cc", "q r s t"] * 2
    direction = torch.linspace(-1, 1, whole.model.config.n_embd)

    def measure(runner):
        start = passes["n"]
        effect = metric_change(lambda logits: logits[0, -1, 3])
        patched = ActivationPatcher(runner).patch_many(
            "hello world", "hellp world", 1, list(range(20)), "positive", effect=effect
        )
        joint = joint_ablation(runner, prompts[:3], 1, 7, "positive", partners=[0, 5])
        lens = transported_direction(runner, prompts, 1, direction, batch_size=2)
        return patched, joint, lens, passes["n"] - start

    a, b = measure(whole), measure(streamed)
    assert b[0] == pytest.approx(a[0], rel=1e-4, abs=1e-6)
    # These KLs are about 1e-6 on the random toy model, so float rounding is most of a
    # difference. reproduce doesn't gate them.
    assert b[1].kl_joint == pytest.approx(a[1].kl_joint, rel=1e-4, abs=1e-7)
    assert b[1].kl_single == pytest.approx(a[1].kl_single, rel=1e-4, abs=1e-7)
    torch.testing.assert_close(b[2], a[2], rtol=1e-4, atol=1e-6)
    assert b[3] < a[3]  # fewer passes, so fewer reads of the streamed layers
