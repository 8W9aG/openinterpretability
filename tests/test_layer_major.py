"""Layer-major passes over a streamed model: the same results, fewer reads of the layers."""

import pytest
import torch

from openinterp.harness import memory
from openinterp.harness.hooks import ablate_pole
from openinterp.harness.layer_major import map_layer_major

TEXTS = [f"text {i} " + "abc" * i for i in range(10)]


@pytest.fixture
def runners(tiny_gpt2, tmp_path, monkeypatch):
    """The tiny GPT-2 loaded whole and streamed from disk."""
    from openinterp.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    tiny_gpt2.save_pretrained(tmp_path / "gpt2")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def load(mode):
        memory.request(mode)
        try:
            model = memory.load_causal_lm(str(tmp_path / "gpt2"), "cpu", torch.float32, None)
        finally:
            memory.request("auto")
        return NeuronHookRunner(model.eval().requires_grad_(False), FakeTokenizer())

    return load("normal"), load("stream")


def _reads(monkeypatch) -> list[str]:
    """Every tensor read from the checkpoint, by name."""
    reads: list[str] = []
    real = memory.OpenCheckpoint.__getitem__

    def counted(self, key):
        reads.append(key)
        return real(self, key)

    monkeypatch.setattr(memory.OpenCheckpoint, "__getitem__", counted)
    return reads


def _two_passes(runner, text: str) -> tuple[float, float]:
    """A pass that stops at layer 1, then a whole pass that ablates on some items only."""
    peak = runner.max_pole_strength(text, 1, 7, "positive")
    ablate = {0: ablate_pole(3, "positive")} if len(text) % 2 else None
    logits = runner.run_with_interventions([text], ablate)
    return peak, float(logits.double().sum())


def test_layer_major_passes_measure_the_same_and_read_fewer_layers(runners, monkeypatch):
    whole, streamed = runners
    expected = [_two_passes(whole, t) for t in TEXTS]
    reads = _reads(monkeypatch)
    in_order = [_two_passes(streamed, t) for t in TEXTS]
    read_in_order = len(reads)
    reads.clear()
    layer_major = map_layer_major(streamed, lambda t: _two_passes(streamed, t), TEXTS)
    # Each pass computes what it computes alone, so not even rounding changes. A hook
    # registered for one item (the ablation) doesn't touch another item's pass.
    assert layer_major == in_order == expected
    assert len(reads) * 4 < read_in_order  # each layer read once per sweep, not per pass
    assert streamed.layer_loads <= 2 * streamed.n_layers  # two sweeps: one per pass


def test_an_item_that_fails_raises_in_the_caller(runners):
    _, streamed = runners

    def fn(text):
        if text == TEXTS[3]:
            raise ValueError("item 3")
        return _two_passes(streamed, text)

    with pytest.raises(ValueError, match="item 3"):
        map_layer_major(streamed, fn, TEXTS)
    # The scheduler is gone: the model streams pass by pass again.
    assert _two_passes(streamed, TEXTS[0]) == _two_passes(runners[0], TEXTS[0])


def test_a_loop_inside_a_running_one_runs_in_order(runners):
    _, streamed = runners

    def outer(text):
        return map_layer_major(streamed, lambda t: _two_passes(streamed, t), [text, text + "z"])

    nested = map_layer_major(streamed, outer, TEXTS[:4])
    whole = runners[0]
    assert nested == [[_two_passes(whole, t), _two_passes(whole, t + "z")] for t in TEXTS[:4]]


def test_a_model_that_fits_runs_the_loop_in_order(runners):
    whole, _ = runners
    assert map_layer_major(whole, lambda t: _two_passes(whole, t), TEXTS) == [
        _two_passes(whole, t) for t in TEXTS
    ]
    assert not hasattr(whole, "layer_loads")


def test_rescoring_a_streamed_scan_reads_each_layer_once(runners, monkeypatch):
    from openinterp.harness import scan_backend

    whole, streamed = runners
    texts = TEXTS * 4  # 40 pieces: three passes of RESCORE_BATCH
    want = scan_backend._measure(whole, 1, 7, "positive")(texts)
    got = scan_backend._measure(streamed, 1, 7, "positive")(texts)
    assert got == want
    assert streamed.layer_loads <= streamed.n_layers  # one sweep for all three passes
