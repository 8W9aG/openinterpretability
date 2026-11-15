"""The CPU thread cap and the fused GELU (issue #3735)."""

import copy

import torch

from openinterp.harness import cpu_speed
from openinterp.harness.environment import run_environment
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.prepare import prepare_model
from tests.conftest import VOCAB, FakeTokenizer


def test_an_oversubscribed_machine_shares_torchs_threads(monkeypatch):
    monkeypatch.delenv(cpu_speed.THREADS_ENV, raising=False)
    # The Linux machine of docs/analyses/cpu-acceleration.md: 4 threads ran fastest.
    assert cpu_speed.cpu_threads(cores=24, load=50, default=16) == 4
    # The Mac at its measured load keeps the default, which was fastest there.
    assert cpu_speed.cpu_threads(cores=10, load=13, default=8) is None
    assert cpu_speed.cpu_threads(cores=4, load=400, default=4) == cpu_speed.MIN_THREADS


def test_the_environment_variable_sets_the_threads(monkeypatch):
    monkeypatch.setenv(cpu_speed.THREADS_ENV, "3")
    assert cpu_speed.cpu_threads(cores=10, load=0, default=8) == 3


def test_the_fused_gelu_computes_what_gpt2s_gelu_does(tiny_gpt2):
    plain = copy.deepcopy(tiny_gpt2)
    prepare_model(tiny_gpt2)
    assert not any(type(m).__name__ == "NewGELUActivation" for m in tiny_gpt2.modules())
    assert sum(isinstance(m, torch.nn.GELU) for m in tiny_gpt2.modules()) == len(
        tiny_gpt2.transformer.h
    )
    ids = torch.randint(0, VOCAB, (2, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        fused, unfused = tiny_gpt2(ids).logits, plain(ids).logits
    assert torch.allclose(fused, unfused, atol=1e-6)
    assert cpu_speed.fuse_gelu(tiny_gpt2) == 0  # nothing left to swap


def test_a_cpu_run_records_its_threads(tiny_gpt2):
    env = run_environment(NeuronHookRunner(tiny_gpt2, FakeTokenizer()))
    assert env.cpu_threads == torch.get_num_threads()
