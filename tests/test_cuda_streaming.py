"""Streaming onto a CUDA GPU (harness/memory.py): the same as the model loaded whole there.

These run only where CUDA is available. CI's runners have no GPU.
"""

import pytest
import torch

from openinterp.harness import memory

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(path, mode, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    memory.request(mode)
    try:
        return memory.load_causal_lm(str(path), "cuda", torch.float32, None).eval()
    finally:
        memory.request("auto")


@pytest.fixture
def saved_gpt2(tiny_gpt2, tmp_path):
    tiny_gpt2.save_pretrained(tmp_path / "gpt2")
    return tmp_path / "gpt2"


def test_a_model_streamed_onto_the_gpu_measures_what_a_loaded_one_does(
    saved_gpt2, monkeypatch, tmp_path
):
    from openinterp.harness.hooks import NeuronHookRunner
    from openinterp.harness.layer_major import map_layer_major
    from openinterp.harness.scan import scan_neuron
    from tests.conftest import FakeTokenizer

    whole = NeuronHookRunner(_load(saved_gpt2, "normal", monkeypatch, tmp_path), FakeTokenizer())
    streamed = NeuronHookRunner(_load(saved_gpt2, "stream", monkeypatch, tmp_path), FakeTokenizer())
    assert memory.is_streamed(streamed.model) and streamed.device.type == "cuda"
    texts = ["abcde", "hello world", "zzzz", "aaaa bbbb"]
    a = scan_neuron(whole, texts, 1, 7, "positive")
    b = scan_neuron(streamed, texts, 1, 7, "positive")
    assert a.examples == b.examples
    torch.testing.assert_close(
        whole.run_with_interventions(texts[:1]), streamed.run_with_interventions(texts[:1])
    )
    peak = lambda runner: lambda t: runner.max_pole_strength(t, 1, 7, "positive")  # noqa: E731
    assert map_layer_major(streamed, peak(streamed), texts) == [peak(whole)(t) for t in texts]


def test_a_moe_streamed_onto_the_gpu_reads_its_slices_there(monkeypatch, tmp_path):
    from tests.test_moe import check

    check.tiny(check.MODELS["Mixtral"]).save_pretrained(tmp_path / "moe")
    whole = _load(tmp_path / "moe", "normal", monkeypatch, tmp_path)
    streamed = _load(tmp_path / "moe", "stream", monkeypatch, tmp_path)
    ids = torch.tensor([[5, 9, 2, 7]], device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(streamed(ids).logits, whole(ids).logits)
