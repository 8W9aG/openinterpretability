"""One heavy job at a time on this machine, when it is turned on (issue #3247).

Loops in different checkouts and sessions share one machine. The device planner
(``openinterp.harness.devices``) moves a job off a busy GPU. That is not enough when every
device is busy, or when the test suite runs beside a scan. With ``OPENINTERP_HEAVY_LOCK=1``,
model loads and ``make test`` take an exclusive lock on one file first, so they queue.

- The file is ``$OPENINTERP_HEAVY_LOCK_FILE``, by default ``~/.cache/openinterp/heavy.lock``.
- The lock is ``fcntl.flock``, because macOS has no ``flock(1)``. It ends when its process
  ends, so a crash can't leave it held.
- A process that loads a model holds the lock until it exits (:func:`hold_for_process`),
  because the model stays in memory until then.
- A holder sets ``OPENINTERP_HEAVY_LOCK_HELD`` to its process id. Its child processes see
  it and skip the lock, so the tests under ``make test`` don't wait for their own parent.

``python -m openinterp.core.heavy_lock <command> …`` runs a command under the lock. The
Makefile's ``test`` target uses it.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ENABLE_ENV = "OPENINTERP_HEAVY_LOCK"  # "1" turns the lock on
FILE_ENV = "OPENINTERP_HEAVY_LOCK_FILE"
HELD_ENV = "OPENINTERP_HEAVY_LOCK_HELD"  # set by a holder for its child processes

_held = None  # this process's open lock file, while it holds the lock


def enabled() -> bool:
    """Whether ``OPENINTERP_HEAVY_LOCK`` turns the lock on."""
    return os.environ.get(ENABLE_ENV, "").lower() in ("1", "true", "yes", "on")


def lock_file() -> Path:
    """``$OPENINTERP_HEAVY_LOCK_FILE``, else ``$XDG_CACHE_HOME/openinterp/heavy.lock``."""
    if value := os.environ.get(FILE_ENV):
        return Path(value)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "openinterp" / "heavy.lock"


def _needed() -> bool:
    """On, not already held by this process, and not held for us by a parent."""
    return enabled() and _held is None and not os.environ.get(HELD_ENV)


def _acquire():
    """Open the lock file and lock it, waiting if another process holds it."""
    path = lock_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or "another process"
        print(f"Waiting for the heavy-job lock ({path}), held by {holder}…", file=sys.stderr)
        fcntl.flock(handle, fcntl.LOCK_EX)
    handle.seek(0)
    handle.truncate()
    handle.write(f"process {os.getpid()}\n")
    handle.flush()
    os.environ[HELD_ENV] = str(os.getpid())
    return handle


def _release(handle) -> None:
    os.environ.pop(HELD_ENV, None)
    handle.close()  # closing the file releases the flock


def hold_for_process() -> None:
    """Take the lock for the rest of this process, if it is on and not yet held.

    The operating system releases it when the process exits.
    """
    global _held
    if _needed():
        _held = _acquire()


@contextmanager
def heavy_job():
    """Hold the lock for a block, if it is on and not yet held."""
    global _held
    if not _needed():
        yield
        return
    _held = _acquire()
    try:
        yield
    finally:
        handle, _held = _held, None
        _release(handle)


def main(argv: list[str]) -> int:
    """Run ``argv`` as a command under the lock, and return its exit code."""
    if not argv:
        print("usage: python -m openinterp.core.heavy_lock <command> [args …]", file=sys.stderr)
        return 2
    with heavy_job():
        return subprocess.run(argv, check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
