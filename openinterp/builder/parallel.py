"""Render many pages in parallel worker processes, where the platform can fork.

Each unit page depends only on the built site tree, so pages can render in any order. A
worker is a fork of the build. So it has the tree already, and nothing large is copied to
it. Each worker renders every ``jobs``-th page and returns the paths that it wrote. The
build then records them in the same order as one process does. With one job, or with no
``fork`` on the platform, the pages render in this process, one by one.

A fork is safe while the build runs no other thread, and the build starts none. If that
changes, Python warns at the fork, and ``openinterp build --jobs 1`` avoids it.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

_WORK: dict[str, Any] = {}  # set just before the workers fork, so that they inherit it


def default_jobs() -> int:
    """Return the number of worker processes to use: one for each CPU."""
    return os.cpu_count() or 1


def render_in_parallel(render, page: Callable[..., None], items: Sequence[tuple]) -> None:
    """Call ``page(*item)`` for each of ``items``, in ``render.jobs`` worker processes.

    ``page`` renders with ``render``, which records each page that it writes.
    """
    jobs = min(render.jobs, len(items))
    if jobs <= 1 or "fork" not in multiprocessing.get_all_start_methods():
        for item in items:
            page(*item)
        return
    _WORK.update(render=render, page=page, items=items)
    try:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(jobs, mp_context=context) as pool:
            shares = list(pool.map(_render_share, range(jobs), [jobs] * jobs))
    finally:
        _WORK.clear()
    # Sort by item only. The sort is stable, so one item's pages keep the order it wrote them.
    written = sorted((pair for share in shares for pair in share), key=lambda pair: pair[0])
    render.report.pages.extend(path for _, path in written)


def _render_share(share: int, jobs: int) -> list[tuple[int, Path]]:
    """Render every ``jobs``-th item, from ``share``, and return each page that it wrote."""
    render, page, items = _WORK["render"], _WORK["page"], _WORK["items"]
    written = []
    for i in range(share, len(items), jobs):
        before = len(render.report.pages)
        page(*items[i])
        written += [(i, path) for path in render.report.pages[before:]]
    return written


def in_background(fn: Callable[[], T], jobs: int) -> Callable[[], T]:
    """Start ``fn()`` in a forked process, and return a function that waits for its result.

    With one job, or with no ``fork`` on the platform, call ``fn()`` now instead. The
    fork uses a plain process and a pipe, not a pool, so the build starts no thread that a
    later fork could copy.
    """
    if jobs <= 1 or "fork" not in multiprocessing.get_all_start_methods():
        value = fn()
        return lambda: value
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_send_result, args=(fn, writer))
    process.start()
    writer.close()

    def result() -> T:
        ok, value = reader.recv()  # before join: a large result fills the pipe
        process.join()
        if not ok:
            raise RuntimeError(f"a background part of the build failed:\n{value}")
        return value

    return result


def _send_result(fn: Callable[[], Any], writer) -> None:
    """Run ``fn()`` in the forked process, and send its result, or the error, to the parent."""
    import traceback

    try:
        writer.send((True, fn()))
    except BaseException:  # noqa: BLE001  (the parent re-raises it)
        writer.send((False, traceback.format_exc()))
    finally:
        writer.close()
