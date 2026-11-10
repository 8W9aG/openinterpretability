"""Read-ahead of the next streamed layer (harness/readahead.py)."""

import torch

from openinterp.harness import memory, readahead
from openinterp.harness.hooks import NeuronHookRunner


def _streamed(tiny_gpt2, tmp_path, monkeypatch):
    tiny_gpt2.save_pretrained(tmp_path / "gpt2")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    memory.request("stream")
    try:
        model = memory.load_causal_lm(str(tmp_path / "gpt2"), "cpu", torch.float32, None)
    finally:
        memory.request("auto")
    from tests.conftest import FakeTokenizer

    return NeuronHookRunner(model.eval(), FakeTokenizer())


def test_each_layer_ranges_cover_its_weights(tiny_gpt2, tmp_path, monkeypatch):
    runner = _streamed(tiny_gpt2, tmp_path, monkeypatch)
    for streamed, loaded in zip(runner.layers, tiny_gpt2.transformer.h, strict=True):
        ranges = readahead.layer_ranges(streamed)
        stored = sum(p.numel() * p.element_size() for p in loaded.parameters())
        assert sum(length for _, _, length in ranges) == stored
        assert len(ranges) == 1  # a layer's tensors sit together in the file


def test_a_pass_requests_each_next_layer(tiny_gpt2, tmp_path, monkeypatch):
    runner = _streamed(tiny_gpt2, tmp_path, monkeypatch)
    wanted = [readahead.layer_ranges(layer) for layer in runner.layers]
    asked = []
    monkeypatch.setattr(readahead, "request", asked.append)
    logits = runner.run_with_interventions(["abcde"])
    n = len(wanted)
    assert asked == [wanted[(i + 1) % n] for i in range(n)]  # the last asks for layer 0
    monkeypatch.undo()
    readahead._read(wanted[0])  # the thread's read, used where posix_fadvise isn't
    torch.testing.assert_close(runner.run_with_interventions(["abcde"]), logits)
