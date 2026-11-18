"""The build's forks run under gc.freeze(), so the workers share the parent's pages (#4592)."""

import gc
import multiprocessing
from types import SimpleNamespace

import pytest

from openinterpretability.builder.parallel import in_background, render_in_parallel

pytestmark = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork"
)


def test_background_fork_is_frozen_and_the_parent_unfrozen():
    before = gc.get_freeze_count()  # the interpreter freezes a few objects at start-up
    result = in_background(gc.get_freeze_count, jobs=2)
    assert gc.get_freeze_count() == 0
    assert result() > before


def test_pool_workers_are_frozen_and_the_parent_unfrozen():
    before = gc.get_freeze_count()
    render = SimpleNamespace(jobs=2, report=SimpleNamespace(pages=[]))

    def page(i):
        render.report.pages.append(f"{i}:{gc.get_freeze_count()}")

    render_in_parallel(render, page, [(0,), (1,)])
    counts = [int(p.split(":")[1]) for p in render.report.pages]
    assert gc.get_freeze_count() == 0
    assert len(counts) == 2 and all(c > before for c in counts)
