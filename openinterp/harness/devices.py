"""Device planner: with ``--device auto`` on a Mac, use Apple's GPU only when it is free.

Two things make the GPU busy. One is another ``openinterp`` process that uses it. Each
process that runs a model on MPS keeps a *lease*. That is an empty file named for its
process id, in ``$OPENINTERP_DEVICE_LEASES`` (default ``~/.cache/openinterp/devices/mps/``).
The planner ignores and removes the lease of a process that stopped. The other is load
from other programs. The planner reads the GPU's ``Device Utilization %`` from ``ioreg``.
It takes the mean of :data:`GPU_SAMPLES` readings, because one reading changes much from
second to second. Above ``$OPENINTERP_GPU_BUSY_PERCENT`` (default
:data:`DEFAULT_GPU_BUSY_PERCENT`), the GPU is busy.

When the GPU is busy, the model runs on the CPU. The CPU is slower than a free GPU. But
two processes on one GPU can slow down each other, and one such run gave wrong results
(docs/experiments.md, Devices). ``--device mps`` uses the GPU even when it is busy.

A process plans once, and every model it loads after that uses the same device.
"""

from __future__ import annotations

import atexit
import os
import re
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from openinterp.core.cache_dirs import cache_root

LEASES_ENV = "OPENINTERP_DEVICE_LEASES"  # lease directory, or "off"
GPU_BUSY_ENV = "OPENINTERP_GPU_BUSY_PERCENT"
# Provisional: replace it with the measured value when docs/experiments.md, Devices, has one.
DEFAULT_GPU_BUSY_PERCENT = 50.0
GPU_SAMPLES = 5
GPU_SAMPLE_SECONDS = 0.4

_UTILIZATION = re.compile(r'"Device Utilization %"=(\d+)')


@dataclass(frozen=True)
class Load:
    """What the planner saw: other processes on each device, and the GPU's utilization."""

    jobs: dict[str, list[int]] = field(default_factory=dict)  # device -> process ids
    gpu_percent: float | None = None  # mean utilization, or None if it was not read


@dataclass(frozen=True)
class DevicePlan:
    """The device to use, and why."""

    device: str
    reason: str
    fallback: bool = False  # True when a busy device was not used


def choose(load: Load, mps_available: bool, busy_percent: float) -> DevicePlan:
    """The device for ``load``: MPS if it is free, else the CPU."""
    if not mps_available:
        return DevicePlan("cpu", "this machine has no Apple GPU")
    others = load.jobs.get("mps", [])
    if others:
        ids = ", ".join(str(pid) for pid in others)
        noun = "job uses" if len(others) == 1 else "jobs use"
        return DevicePlan("cpu", f"another openinterp {noun} the GPU (process {ids})", True)
    if load.gpu_percent is not None and load.gpu_percent > busy_percent:
        reason = f"the GPU is {load.gpu_percent:.0f}% busy, more than {busy_percent:.0f}%"
        return DevicePlan("cpu", reason, True)
    return DevicePlan("mps", "the GPU is free")


def busy_percent() -> float:
    """``$OPENINTERP_GPU_BUSY_PERCENT``, else :data:`DEFAULT_GPU_BUSY_PERCENT`."""
    return float(os.environ.get(GPU_BUSY_ENV) or DEFAULT_GPU_BUSY_PERCENT)


def read_load() -> Load:
    """The other processes on MPS, and the GPU's utilization if no other process is on it."""
    jobs = {"mps": holders("mps")}
    return Load(jobs, None if jobs["mps"] else gpu_utilization())


def gpu_utilization(samples: int = GPU_SAMPLES, seconds: float = GPU_SAMPLE_SECONDS):
    """The mean of ``samples`` readings of the GPU's utilization, or None if ``ioreg`` fails."""
    readings = []
    for i in range(samples):
        if i:
            time.sleep(seconds)
        value = _read_utilization()
        if value is None:
            return None
        readings.append(value)
    return sum(readings) / len(readings)


def _read_utilization() -> float | None:
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_utilization(out)


def parse_utilization(ioreg_output: str) -> float | None:
    """The largest ``Device Utilization %`` in ``ioreg`` output, or None if there is none."""
    values = [float(v) for v in _UTILIZATION.findall(ioreg_output)]
    return max(values) if values else None


def lease_root() -> Path | None:
    """``$OPENINTERP_DEVICE_LEASES``, else ``$XDG_CACHE_HOME/openinterp/devices``; None if off."""
    return cache_root(LEASES_ENV, "devices")


def holders(device: str) -> list[int]:
    """Process ids of the other running processes with a lease on ``device``.

    It removes the leases of processes that have stopped.
    """
    root = lease_root()
    folder = root / device if root else None
    if folder is None or not folder.is_dir():
        return []
    found = []
    for lease in folder.iterdir():
        if not lease.name.isdigit() or int(lease.name) == os.getpid():
            continue
        if _running(int(lease.name)):
            found.append(int(lease.name))
        else:
            with suppress(OSError):
                lease.unlink()
    return sorted(found)


def take_lease(device: str) -> None:
    """Record that this process uses ``device``, until it stops."""
    root = lease_root()
    if root is None:
        return
    lease = root / device / str(os.getpid())
    if lease.exists():
        return
    with suppress(OSError):
        lease.parent.mkdir(parents=True, exist_ok=True)
        lease.touch()
        atexit.register(_drop, lease)


def _drop(lease: Path) -> None:
    with suppress(OSError):
        lease.unlink()


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it runs, as another user
        return True
    return True


_planned: DevicePlan | None = None
_noticed = False


def notice() -> str | None:
    """At the first call only: why the plan did not use a busy GPU. Else None."""
    global _noticed
    if _noticed or _planned is None or not _planned.fallback:
        return None
    _noticed = True
    return f"Using the CPU, because {_planned.reason}. Pass --device mps to share the GPU."


def planned() -> DevicePlan:
    """This process's plan, made at the first call."""
    global _planned
    if _planned is None:
        import torch

        _planned = choose(read_load(), torch.backends.mps.is_available(), busy_percent())
    return _planned


def forget() -> None:
    """Make the next :func:`planned` plan again (for tests)."""
    global _planned, _noticed
    _planned, _noticed = None, False
