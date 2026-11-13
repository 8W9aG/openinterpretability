"""Device planning on a machine with CUDA GPUs: the first free GPU, else the CPU.

A GPU is busy when another ``openinterp`` process has a lease on it (``cuda0``,
``cuda1``, … in the lease folder of ``harness/devices.py``). It is also busy when its
utilization is more than the planner's threshold. That is the mean of
:data:`~openinterp.harness.devices.GPU_SAMPLES` readings of ``nvidia-smi``. If every
GPU is busy, the job runs on the CPU. A machine with one GPU uses ``cuda``, and a
machine with more uses ``cuda:<index>``.
"""

from __future__ import annotations

import subprocess
import time

from openinterp.harness.devices import (
    GPU_SAMPLE_SECONDS,
    GPU_SAMPLES,
    DevicePlan,
    holders,
)

NVIDIA_SMI = ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"]


def parse_nvidia_smi(output: str) -> dict[int, float]:
    """Each GPU's utilization from ``nvidia-smi`` output, by index. Bad lines are skipped."""
    found = {}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                found[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return found


def _read() -> dict[int, float] | None:
    try:
        out = subprocess.run(NVIDIA_SMI, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(out.stdout) or None


def utilization(samples: int = GPU_SAMPLES, seconds: float = GPU_SAMPLE_SECONDS):
    """Each GPU's mean utilization over ``samples`` readings, or None if it can't be read."""
    readings: list[dict[int, float]] = []
    for i in range(samples):
        if i:
            time.sleep(seconds)
        reading = _read()
        if reading is None:
            return None
        readings.append(reading)
    gpus = set().union(*readings)
    return {g: sum(r.get(g, 0.0) for r in readings) / len(readings) for g in gpus}


def choose_cuda(
    count: int, jobs: dict[int, list[int]], percents: dict[int, float] | None, busy: float
) -> DevicePlan:
    """The first of ``count`` GPUs that is free, else the CPU."""
    reasons = []
    for gpu in range(count):
        others = jobs.get(gpu, [])
        percent = (percents or {}).get(gpu)
        if others:
            ids = ", ".join(str(pid) for pid in others)
            reasons.append(f"GPU {gpu} is in use by another openinterp job (process {ids})")
        elif percent is not None and percent > busy:
            reasons.append(f"GPU {gpu} is {percent:.0f}% busy, more than {busy:.0f}%")
        else:
            device = "cuda" if count == 1 else f"cuda:{gpu}"
            return DevicePlan(device, f"GPU {gpu} is free", hint="cuda")
    reason = "; ".join(reasons) or "this machine has no CUDA GPU"
    return DevicePlan("cpu", reason, fallback=count > 0, hint="cuda")


def plan_cuda(busy: float) -> DevicePlan:
    """The plan on this machine's CUDA GPUs."""
    import torch

    count = torch.cuda.device_count()
    jobs = {gpu: holders(f"cuda{gpu}") for gpu in range(count)}
    free = [gpu for gpu in range(count) if not jobs[gpu]]
    return choose_cuda(count, jobs, utilization() if free else None, busy)
