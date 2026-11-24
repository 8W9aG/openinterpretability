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


def test_a_build_in_the_suite_takes_one_job_unless_it_asks_for_more(monkeypatch):  # #6507
    from openinterpretability.builder.parallel import JOBS_ENV, default_jobs

    assert default_jobs() == 1  # tests/conftest.py sets it
    monkeypatch.setenv(JOBS_ENV, "3")
    assert default_jobs() == 3
    monkeypatch.delenv(JOBS_ENV)
    assert default_jobs() >= 1


def test_with_no_cap_a_build_takes_one_job_per_cpu(monkeypatch):
    from openinterpretability.builder import parallel

    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 8)
    assert parallel.default_jobs({}) == 8 and parallel.default_jobs({parallel.JOBS_ENV: ""}) == 8
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: None)  # unknown: one job
    assert parallel.default_jobs({}) == 1
