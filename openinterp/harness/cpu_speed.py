"""Two ways to run CPU forward passes faster within ``reproduce``'s tolerance (issue #3735).

``docs/analyses/cpu-acceleration.md`` measured both.

* **Fewer threads on a shared machine.** Several loops on one machine each start torch's
  default thread count, so their threads compete. On a Linux machine at load 50 on 24
  threads, 4 threads ran a forward pass twice as fast as the default 16. On a Mac at a
  light load, the default was fastest. So :func:`cpu_threads` caps the threads only when
  the machine is oversubscribed. MKL's sums depend on the thread count, so results move by
  up to 1.5e-5 of the max. The record keeps the count (``RunEnvironment.cpu_threads``).
* **The fused GELU.** GPT-2's ``NewGELUActivation`` is written in Python, so it runs as
  five kernels. :func:`fuse_gelu` swaps it for ``torch.nn.GELU(approximate="tanh")``, the
  same formula as one kernel. That made forward passes 1.12 to 1.27 times as fast. On the
  Mac CPU it is identical bit for bit. On AVX2 and MPS it differs by up to 1e-6.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn

THREADS_ENV = "OPENINTERP_CPU_THREADS"
# The 1-minute load average, over the cores, above which the threads are capped.
OVERSUBSCRIBED = 1.5
MIN_THREADS = 2


def cpu_threads(cores: int, load: float, default: int) -> int | None:
    """The thread count for a CPU run, or None to keep torch's ``default``.

    ``$OPENINTERP_CPU_THREADS`` wins when set. Otherwise, at a load above
    :data:`OVERSUBSCRIBED` times the ``cores``, each of about ``load / default`` jobs gets
    an equal share of ``default``.
    """
    if forced := os.environ.get(THREADS_ENV):
        return max(1, int(forced))
    if load <= OVERSUBSCRIBED * cores:
        return None
    return max(MIN_THREADS, default // math.ceil(load / default))


def apply_cpu_threads() -> int | None:
    """Set torch's threads by :func:`cpu_threads` for this machine; return the count set."""
    threads = cpu_threads(os.cpu_count() or 1, os.getloadavg()[0], torch.get_num_threads())
    if threads is not None and threads != torch.get_num_threads():
        torch.set_num_threads(threads)
        return threads
    return None


def fuse_gelu(model: nn.Module) -> int:
    """Swap each ``NewGELUActivation`` for torch's fused tanh GELU; return how many."""
    swapped = 0
    for parent in model.modules():
        for name, child in parent.named_children():
            if type(child).__name__ == "NewGELUActivation":
                setattr(parent, name, nn.GELU(approximate="tanh"))
                swapped += 1
    return swapped
