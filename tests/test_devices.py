"""Device choice: ``--device auto`` picks CUDA, then Apple's GPU, then the CPU."""

import copy

import pytest
import torch

from openinterp.cli import resolve_device
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.scan import scan_neuron
from openinterp.reproduce import DEFAULT_TOLERANCE
from tests.conftest import FakeTokenizer
from tests.test_agents import TEXTS


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"), [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_auto_prefers_cuda_then_mps(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device("auto") == expected
    assert resolve_device("cpu") == "cpu"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple's GPU")
def test_an_mps_scan_reproduces_on_the_cpu(tiny_gpt2):
    cpu = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    mps = NeuronHookRunner(copy.deepcopy(tiny_gpt2).to("mps"), FakeTokenizer())
    texts = [t * 3 for t in TEXTS]
    on_cpu = scan_neuron(cpu, texts, 1, 2, "positive", batch_tokens=64)
    on_mps = scan_neuron(mps, texts, 1, 2, "positive", batch_tokens=64)
    scale = max(on_cpu.values)
    pairs = zip(on_mps.values, on_cpu.values, strict=True)
    assert all(DEFAULT_TOLERANCE.close(m, c, scale) for m, c in pairs)
    assert [ex.text for ex in on_mps.examples] == [ex.text for ex in on_cpu.examples]
