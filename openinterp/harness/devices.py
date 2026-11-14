"""Device planner: with ``--device auto``, use a GPU only when it is free.

This module plans for Apple's GPU (MPS). ``cuda_devices.py`` plans for CUDA GPUs with
the same leases and threshold, and chooses the first free GPU.

Two things make the GPU busy. One is another ``open-interpretability`` process that uses it. Each
process that runs a model on MPS keeps a *lease*. That is an empty file named for its
process id, in ``$OPENINTERP_DEVICE_LEASES`` (default ``~/.cache/openinterp/devices/mps/``).
The planner ignores and removes the lease of a process that stopped. The other is load
from other programs. The planner reads the GPU's ``Device Utilization %`` from ``ioreg``.
It takes the mean of :data:`GPU_SAMPLES` readings, because one reading changes much from
second to second. Above ``$OPENINTERP_GPU_BUSY_PERCENT`` (default
:data:`DEFAULT_GPU_BUSY_PERCENT`), the GPU is busy.

When the GPU is busy, the order is MPS, then the ANE, then the CPU. The ANE is Apple's
Neural Engine. If this Mac can run Core ML and no other job has a lease on the ANE, the
scans' forward passes run there (``harness/ane/``). The rest runs on the CPU. Else everything runs
on the CPU. The CPU is slower than a free GPU. But two processes on one GPU can slow down
each other, and one such run gave wrong results (docs/experiments.md, Devices).
``--device mps`` uses the GPU even when it is busy. ``--device ane`` runs the scans on
the ANE, and the rest where the planner puts it.

A model that streams (``--memory stream``) is the exception. Its scan on the CPU reads
each layer once for the whole corpus, and with the CPU free that is faster than the ANE.
On a 50,000-token Qwen2.5-0.5B scan, the CPU's pass took about 16 s. The ANE's took
39 s, and it needed 40 s more to measure the top pieces again. So a streamed model's
scans use the ANE only if the CPU is busy too. That is a load average of at least
``$OPENINTERP_CPU_BUSY_PERCENT`` (default :data:`DEFAULT_CPU_BUSY_PERCENT`) of its cores.

A process plans once, and every model it loads after that uses the same device.
"""

from __future__ import annotations

import atexit
import os
import platform
import re
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from openinterp.core.cache_dirs import cache_root

LEASES_ENV = "OPENINTERP_DEVICE_LEASES"  # lease directory, or "off"
# Set by the retry after an MPS failure: every model of the command loads on this device.
FORCE_ENV = "OPENINTERP_DEVICE_FORCE"
GPU_BUSY_ENV = "OPENINTERP_GPU_BUSY_PERCENT"
# Idle reads 1%, one job 93-97% (docs/analyses/device-contention.md).
DEFAULT_GPU_BUSY_PERCENT = 50.0
CPU_BUSY_ENV = "OPENINTERP_CPU_BUSY_PERCENT"
DEFAULT_CPU_BUSY_PERCENT = 50.0  # the 1-minute load average, as a share of the cores
# The longest the first operation on MPS may take before the process uses the CPU.
START_ENV = "OPENINTERP_MPS_START_SECONDS"
DEFAULT_START_SECONDS = 120.0
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
    scan: str | None = None  # "ane" when scans run on Apple's Neural Engine
    hint: str = "mps"  # the --device value that uses the GPU even when it is busy


def choose(
    load: Load, mps_available: bool, busy_percent: float, ane_ok: bool = False
) -> DevicePlan:
    """The device for ``load``: MPS if it is free, else the ANE for scans, else the CPU.

    With the ANE, the scans' forward passes run there, and the rest runs on the CPU.
    ``ane_ok`` says that this machine can run Core ML on the ANE.
    """
    if not mps_available:
        return DevicePlan("cpu", "this machine has no Apple GPU")
    busy = _busy(load.jobs.get("mps", []), "the GPU", load.gpu_percent, busy_percent)
    if busy is None:
        return DevicePlan("mps", "the GPU is free")
    if ane_ok and not load.jobs.get("ane"):
        return DevicePlan("cpu", busy, True, scan="ane")
    if ane_ok:
        busy += " and " + _busy(load.jobs["ane"], "the ANE", None, busy_percent)
    return DevicePlan("cpu", busy, True)


def _busy(others: list[int], name: str, percent: float | None, limit: float) -> str | None:
    """Why ``name`` is busy, or None if it is free."""
    if others:
        ids = ", ".join(str(pid) for pid in others)
        noun = "job uses" if len(others) == 1 else "jobs use"
        return f"another openinterp {noun} {name} (process {ids})"
    if percent is not None and percent > limit:
        return f"{name} is {percent:.0f}% busy, more than {limit:.0f}%"
    return None


def busy_percent() -> float:
    """``$OPENINTERP_GPU_BUSY_PERCENT``, else :data:`DEFAULT_GPU_BUSY_PERCENT`."""
    return float(os.environ.get(GPU_BUSY_ENV) or DEFAULT_GPU_BUSY_PERCENT)


def read_load() -> Load:
    """The other processes on MPS and the ANE, and the GPU's utilization if no job is on it."""
    jobs = {"mps": holders("mps"), "ane": holders("ane")}
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


_used: set[str] = set()  # devices that this process ran models on


def lease_name(device: str) -> str | None:
    """The lease folder of a GPU device (``mps``, ``cuda0``, ``cuda1``, …), or None."""
    if device == "mps":
        return "mps"
    if device.startswith("cuda"):
        return f"cuda{device.partition(':')[2] or 0}"
    return None


def take_lease(device: str) -> None:
    """Record that this process uses the GPU ``device``, until it stops."""
    name = lease_name(device)
    if name is None:
        return
    _used.add(name)
    root = lease_root()
    if root is None:
        return
    lease = root / name / str(os.getpid())
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


_choice: str | None = None  # how this process chose its device, for the record


def note_choice(text: str) -> None:
    """Record how this process chose its device (``RunEnvironment.device_choice``)."""
    global _choice
    _choice = text


def choice() -> str | None:
    """How this process chose its device, or None if nothing chose it (e.g. in a test)."""
    return _choice


_planned: DevicePlan | None = None
_noticed = False
_forced_ane = False


def force_ane() -> None:
    """Run scans on the ANE whatever the plan says (``--device ane``)."""
    global _forced_ane
    _forced_ane = True


def cpu_load_percent() -> float:
    """The 1-minute load average, as a percentage of the CPU's cores."""
    return 100.0 * os.getloadavg()[0] / (os.cpu_count() or 1)


def streamed_scans_skip_ane() -> str | None:
    """Why a streamed model's scans run on the CPU and not on the planned ANE, or None.

    The CPU's scan of a streamed model is faster than the ANE's when the CPU is free.
    ``--device ane`` keeps the ANE.
    """
    if _forced_ane:
        return None
    load = cpu_load_percent()
    if load >= float(os.environ.get(CPU_BUSY_ENV) or DEFAULT_CPU_BUSY_PERCENT):
        return None
    return f"the model streams and the CPU is free (load {load:.0f}% of its cores)"


def scans_on_ane() -> bool:
    """True if scans should run on the ANE: ``--device ane``, or the plan's choice."""
    return _forced_ane or (_planned is not None and _planned.scan == "ane")


def notice() -> str | None:
    """At the first call only: why the plan did not use a busy GPU. Else None."""
    global _noticed
    if _noticed or _planned is None or not _planned.fallback:
        return None
    _noticed = True
    ane = " Scans run on Apple's Neural Engine." if _planned.scan == "ane" else ""
    return (
        f"Using the CPU, because {_planned.reason}.{ane} "
        f"Pass --device {_planned.hint} to share the GPU."
    )


def planned() -> DevicePlan:
    """This process's plan, made at the first call."""
    global _planned
    if _planned is None:
        import torch

        if torch.cuda.is_available():
            from openinterp.harness.cuda_devices import plan_cuda

            _planned = plan_cuda(busy_percent())
        else:
            mps = torch.backends.mps.is_available()
            _planned = choose(read_load(), mps, busy_percent(), mps and ane_possible())
    return _planned


def ane_possible() -> bool:
    """True if this machine can run Core ML models on the ANE: a Mac with coremltools."""
    from openinterp.harness.ane import convert

    return platform.system() == "Darwin" and platform.machine() == "arm64" and convert.available()


def forget() -> None:
    """Make the next :func:`planned` plan again (for tests)."""
    global _planned, _noticed, _forced_ane, _start_checked, _choice
    _planned, _noticed, _forced_ane, _start_checked, _choice = None, False, False, False, None
    _used.clear()


def forced() -> str | None:
    """The device that a retry after a GPU failure set, or None."""
    return os.environ.get(FORCE_ENV) or None


def gpu_failed(exc: BaseException) -> str | None:
    """``MPS`` or ``CUDA`` if ``exc`` is a GPU failure that a retry on the CPU can get past.

    It is one if this process ran a model on that GPU and an exception in the chain
    names it. A command that is a retry already is not retried. Else None.
    """
    if forced():
        return None
    kinds = {"mps"} & _used | ({"cuda"} if any(u.startswith("cuda") for u in _used) else set())
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc).lower() if isinstance(exc, Exception) else ""
        if found := next((k for k in sorted(kinds) if k in text), None):
            return found.upper()
        exc = exc.__cause__ or exc.__context__
    return None


def rerun_on_cpu(exc: BaseException, gpu: str = "MPS") -> None:
    """Run this command again from the start, in a new process, with every model on the CPU.

    The new process has none of the GPU's state. Files that the first run wrote are
    written again.
    """
    import sys

    first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
    print(
        f"The GPU ({gpu}) failed ({first}). Running the command again on the CPU.", file=sys.stderr
    )
    sys.stderr.flush()
    sys.stdout.flush()
    # The same command line that started this process, so nothing new runs.
    os.execve(sys.executable, sys.orig_argv, {**os.environ, FORCE_ENV: "cpu"})  # noqa: S606


_start_checked: str | None | bool = False  # False: not checked yet


def mps_stuck() -> str | None:
    """Why this process can't use MPS, or None if a first operation on it ends in time.

    Several processes that start on MPS at the same moment can each stop for good in
    Metal's compiler. It opens Metal's shared cache of compiled programs on disk
    (docs/analyses/device-contention.md). So a first small operation runs on a
    background thread. The process waits for it at most ``$OPENINTERP_MPS_START_SECONDS``
    (default :data:`DEFAULT_START_SECONDS`). The check runs once in a process.
    """
    global _start_checked
    if _start_checked is False:
        _start_checked = _probe_mps(float(os.environ.get(START_ENV) or DEFAULT_START_SECONDS))
    return _start_checked


def _probe_mps(seconds: float) -> str | None:
    done, failed = threading.Event(), []

    def probe() -> None:
        try:
            import torch

            torch.ones(1024, device="mps").mul(2).cpu()
            torch.mps.synchronize()
        except Exception as exc:  # a GPU that fails at once can't be used
            failed.append(exc)
        finally:
            done.set()

    threading.Thread(target=probe, daemon=True, name="mps-start").start()
    if not done.wait(timeout=seconds):
        return f"the GPU (MPS) didn't start in {seconds:g} s (${START_ENV})"
    if failed:
        return f"the GPU (MPS) failed at once ({type(failed[0]).__name__}: {failed[0]})"
    return None
