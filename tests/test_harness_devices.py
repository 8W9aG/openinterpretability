"""Device planner helpers: ioreg readings, lease clean-up, process checks, CPU load, MPS start."""

import subprocess
from types import SimpleNamespace

import torch

from openinterpretability.harness import devices

REAL_GPU_UTILIZATION = devices.gpu_utilization  # conftest replaces these for every test
REAL_MPS_STUCK = devices.mps_stuck

IOREG = '"PerformanceStatistics" = {"Device Utilization %"=42}'


def test_one_failed_reading_makes_the_utilization_unknown(monkeypatch):
    readings = iter([30.0, None])
    sleeps = []
    monkeypatch.setattr(devices, "_read_utilization", lambda: next(readings))
    monkeypatch.setattr(devices.time, "sleep", sleeps.append)
    assert REAL_GPU_UTILIZATION(3, 0.25) is None
    assert sleeps == [0.25]  # it waited once, between the two readings


def test_the_utilization_is_read_from_ioreg(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=IOREG)

    monkeypatch.setattr(devices.subprocess, "run", run)
    assert devices._read_utilization() == 42.0
    assert calls == [["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"]]


def test_a_missing_or_failing_ioreg_gives_no_utilization(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("ioreg")

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired("ioreg", 5)

    for fail in (missing, slow):
        monkeypatch.setattr(devices.subprocess, "run", fail)
        assert devices._read_utilization() is None


def test_dropping_a_lease_removes_it_and_ignores_a_missing_one(tmp_path):
    lease = tmp_path / "mps" / "123"
    lease.parent.mkdir()
    lease.touch()
    devices._drop(lease)
    assert not lease.exists()
    devices._drop(lease)  # already gone: no error


def test_a_process_of_another_user_counts_as_running(monkeypatch):
    def denied(pid, signal):
        raise PermissionError(pid)

    monkeypatch.setattr(devices.os, "kill", denied)
    assert devices._running(1) is True


def test_cpu_load_is_the_load_average_over_the_cores(monkeypatch):
    monkeypatch.setattr(devices.os, "getloadavg", lambda: (3.0, 2.0, 1.0))
    monkeypatch.setattr(devices.os, "cpu_count", lambda: 4)
    assert devices.cpu_load_percent() == 75.0
    monkeypatch.setattr(devices.os, "cpu_count", lambda: None)  # unknown: one core
    assert devices.cpu_load_percent() == 300.0


def test_a_first_mps_operation_that_ends_in_time_is_no_problem(monkeypatch):
    # Run the probe's operation on the CPU, as a working GPU would run it.
    real_ones = torch.ones
    monkeypatch.setattr(torch, "ones", lambda *a, device=None, **k: real_ones(*a, **k))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    assert devices._probe_mps(30) is None


def test_the_mps_start_check_runs_once_in_a_process(monkeypatch):
    calls = []

    def probe(seconds):
        calls.append(seconds)
        return "the GPU (MPS) didn't start in 7 s"

    monkeypatch.setattr(devices, "mps_stuck", REAL_MPS_STUCK)
    monkeypatch.setattr(devices, "_start_checked", False)
    monkeypatch.setattr(devices, "_probe_mps", probe)
    monkeypatch.setenv(devices.START_ENV, "7")
    assert devices.mps_stuck() == "the GPU (MPS) didn't start in 7 s"
    assert devices.mps_stuck() == "the GPU (MPS) didn't start in 7 s"  # the first answer
    assert calls == [7.0]
