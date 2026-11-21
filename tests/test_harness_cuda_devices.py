"""CUDA GPU readings from nvidia-smi, and their mean over samples (harness/cuda_devices.py)."""

import subprocess
from types import SimpleNamespace

from openinterpretability.harness import cuda_devices

REAL_UTILIZATION = cuda_devices.utilization  # conftest replaces it for every test


def test_a_gpu_with_no_number_is_skipped():
    assert cuda_devices.parse_nvidia_smi("0, [N/A]\n1, 12\n") == {1: 12.0}


def test_nvidia_smi_is_run_and_read(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="0, 7\n1, 60\n")

    monkeypatch.setattr(cuda_devices.subprocess, "run", run)
    assert cuda_devices._read() == {0: 7.0, 1: 60.0}
    assert calls == [cuda_devices.NVIDIA_SMI]


def test_no_gpu_lines_or_a_failing_nvidia_smi_give_no_reading(monkeypatch):
    monkeypatch.setattr(
        cuda_devices.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="No devices")
    )
    assert cuda_devices._read() is None

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(9, args[0])

    monkeypatch.setattr(cuda_devices.subprocess, "run", fail)
    assert cuda_devices._read() is None


def test_utilization_is_each_gpus_mean_over_the_samples(monkeypatch):
    readings = iter([{0: 10.0, 1: 90.0}, {0: 30.0}])  # GPU 1 is missing from one reading
    sleeps = []
    monkeypatch.setattr(cuda_devices, "_read", lambda: next(readings))
    monkeypatch.setattr(cuda_devices.time, "sleep", sleeps.append)
    assert REAL_UTILIZATION(2, 0.5) == {0: 20.0, 1: 45.0}
    assert sleeps == [0.5]


def test_one_failed_reading_makes_the_utilization_unknown(monkeypatch):
    readings = iter([{0: 10.0}, None])
    monkeypatch.setattr(cuda_devices, "_read", lambda: next(readings))
    monkeypatch.setattr(cuda_devices.time, "sleep", lambda seconds: None)
    assert REAL_UTILIZATION(3, 0) is None
