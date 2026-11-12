"""Device choice: ``--device auto`` picks CUDA, then Apple's GPU if it is free, then the CPU."""

import copy
import os
import subprocess
import sys

import pytest
import torch

from openinterp.cli import resolve_device
from openinterp.harness import devices
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.scan import scan_neuron
from openinterp.reproduce import DEFAULT_TOLERANCE
from tests.conftest import FakeTokenizer
from tests.test_agents import TEXTS

REAL_GPU_UTILIZATION = devices.gpu_utilization  # conftest replaces it for every test


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"), [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_auto_prefers_cuda_then_mps(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device("auto") == expected
    assert resolve_device("cpu") == "cpu"


def test_the_planner_uses_a_free_gpu():
    plan = devices.choose(devices.Load({"mps": []}, 3.0), True, 50.0)
    assert (plan.device, plan.fallback) == ("mps", False)


def test_another_openinterp_job_on_the_gpu_sends_this_one_to_the_cpu():
    plan = devices.choose(devices.Load({"mps": [41, 42]}, None), True, 50.0)
    assert (plan.device, plan.fallback) == ("cpu", True)
    assert "41, 42" in plan.reason


def test_a_busy_gpu_sends_the_job_to_the_cpu():
    assert devices.choose(devices.Load({"mps": []}, 51.0), True, 50.0).device == "cpu"
    assert devices.choose(devices.Load({"mps": []}, 50.0), True, 50.0).device == "mps"


def test_no_apple_gpu_means_the_cpu_without_a_fallback_note():
    plan = devices.choose(devices.Load(), False, 50.0)
    assert (plan.device, plan.fallback) == ("cpu", False)


def test_an_unreadable_utilization_does_not_count_as_busy():
    assert devices.choose(devices.Load({"mps": []}, None), True, 50.0).device == "mps"


def test_utilization_is_read_from_ioreg_output():
    out = '"PerformanceStatistics" = {"Tiler Utilization %"=53,"Device Utilization %"=84}'
    assert devices.parse_utilization(out) == 84.0
    assert devices.parse_utilization("nothing here") is None


def test_the_utilization_is_a_mean_of_samples(monkeypatch):
    readings = iter([0.0, 100.0, 50.0])
    monkeypatch.setattr(devices, "_read_utilization", lambda: next(readings))
    assert REAL_GPU_UTILIZATION(3, 0) == 50.0


def test_leases_of_running_processes_count_and_stopped_ones_go():
    folder = devices.lease_root() / "mps"
    folder.mkdir(parents=True)
    parent = os.getppid()  # running
    stopped = subprocess.Popen([sys.executable, "-c", "pass"])
    stopped.wait()
    for pid in (parent, stopped.pid, os.getpid()):
        (folder / str(pid)).touch()
    assert devices.holders("mps") == [parent]  # not this process, not the stopped one
    assert not (folder / str(stopped.pid)).exists()


def test_a_process_on_mps_takes_a_lease_that_others_see():
    devices.take_lease("mps")
    assert (devices.lease_root() / "mps" / str(os.getpid())).exists()
    assert devices.holders("mps") == []  # a process does not count its own lease


def test_auto_falls_back_once_with_a_note(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(devices, "holders", lambda device: [7])
    assert resolve_device("auto") == "cpu"
    assert resolve_device("auto") == "cpu"
    assert capsys.readouterr().err.count("Using the CPU, because another openinterp job") == 1


def test_leases_can_be_turned_off(monkeypatch):
    monkeypatch.setenv(devices.LEASES_ENV, "off")
    devices.take_lease("mps")
    assert devices.holders("mps") == []


def _oom():
    try:
        raise RuntimeError("MPS backend out of memory (MPS allocated: 40 GB)")
    except RuntimeError as inner:
        raise ValueError("the scan failed") from inner


def test_an_mps_error_in_a_process_on_mps_is_retried():
    devices.take_lease("mps")
    with pytest.raises(ValueError) as caught:
        _oom()
    assert devices.gpu_failed(caught.value)  # found down the chain
    assert not devices.gpu_failed(RuntimeError("index out of range"))


def test_no_retry_without_mps_or_twice(monkeypatch):
    with pytest.raises(ValueError) as caught:
        _oom()
    assert not devices.gpu_failed(caught.value)  # this process never ran on MPS
    devices.take_lease("mps")
    monkeypatch.setenv(devices.FORCE_ENV, "cpu")
    assert not devices.gpu_failed(caught.value)  # a retry already


def test_a_retry_forces_every_model_onto_the_cpu(monkeypatch):
    monkeypatch.setenv(devices.FORCE_ENV, "cpu")
    assert resolve_device("auto") == "cpu" and resolve_device("mps") == "cpu"


def test_main_runs_the_command_again_on_the_cpu(monkeypatch, capsys):
    import openinterp.cli as cli_module

    def failing(**_):
        devices.take_lease("mps")
        _oom()

    calls = []
    monkeypatch.setattr(cli_module, "cli", failing)
    monkeypatch.setattr(
        os, "execve", lambda *a: calls.append(a) or (_ for _ in ()).throw(SystemExit(0))
    )
    with pytest.raises(SystemExit):
        cli_module.main()
    (exe, argv, env) = calls[0]
    assert exe == sys.executable and argv == sys.orig_argv and env[devices.FORCE_ENV] == "cpu"
    assert "Running the command again on the CPU" in capsys.readouterr().err


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


# -- CUDA --------------------------------------------------------------------


def test_nvidia_smi_output_gives_each_gpus_utilization():
    from openinterp.harness.cuda_devices import parse_nvidia_smi

    assert parse_nvidia_smi("0, 3\n1, 97\n") == {0: 3.0, 1: 97.0}
    assert parse_nvidia_smi("No devices were found") == {}


def test_the_first_free_cuda_gpu_is_chosen():
    from openinterp.harness.cuda_devices import choose_cuda

    plan = choose_cuda(3, {0: [11]}, {1: 95.0, 2: 4.0}, 50.0)
    assert (plan.device, plan.fallback) == ("cuda:2", False)
    assert choose_cuda(1, {}, {0: 2.0}, 50.0).device == "cuda"  # one GPU keeps "cuda"


def test_every_busy_cuda_gpu_sends_the_job_to_the_cpu_with_a_cuda_hint():
    from openinterp.harness.cuda_devices import choose_cuda

    plan = choose_cuda(2, {0: [11]}, {1: 99.0}, 50.0)
    assert (plan.device, plan.fallback, plan.hint) == ("cpu", True, "cuda")
    assert "GPU 0 is in use" in plan.reason and "GPU 1 is 99% busy" in plan.reason


def test_auto_on_cuda_skips_a_gpu_another_job_holds(monkeypatch, capsys):
    from openinterp.harness import cuda_devices

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(cuda_devices, "holders", lambda name: [7] if name == "cuda0" else [])
    assert resolve_device("auto") == "cuda:1"
    assert (devices.lease_root() / "cuda1" / str(os.getpid())).exists()


def test_cuda_leases_are_named_by_gpu():
    assert devices.lease_name("cuda") == "cuda0" and devices.lease_name("cuda:3") == "cuda3"
    assert devices.lease_name("mps") == "mps" and devices.lease_name("cpu") is None


def test_a_cuda_out_of_memory_error_is_retried_on_the_cpu():
    devices.take_lease("cuda:1")
    assert devices.gpu_failed(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) == "CUDA"
    assert devices.gpu_failed(RuntimeError("MPS backend out of memory")) is None  # not used
