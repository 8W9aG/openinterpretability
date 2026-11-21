"""The parallel build's worker functions, run in this process, and a failed background part."""

import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from openinterpretability.builder import parallel


def test_a_worker_renders_every_jobs_th_item_and_returns_what_it_wrote(monkeypatch):
    render = SimpleNamespace(report=SimpleNamespace(pages=[]))

    def page(name, copies):
        render.report.pages.extend(Path(f"{name}-{i}.html") for i in range(copies))

    items = [("a", 1), ("b", 2), ("c", 0), ("d", 2)]
    monkeypatch.setitem(parallel._WORK, "render", render)
    monkeypatch.setitem(parallel._WORK, "page", page)
    monkeypatch.setitem(parallel._WORK, "items", items)
    # Worker 2 of 2 takes items 1 and 3. Item 1 writes two pages, in its own order.
    assert parallel._render_share(1, 2) == [
        (1, Path("b-0.html")),
        (1, Path("b-1.html")),
        (3, Path("d-0.html")),
        (3, Path("d-1.html")),
    ]


class _Writer:
    def __init__(self):
        self.sent, self.closed = [], False

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


def test_a_background_part_sends_its_result_and_closes_the_pipe():
    writer = _Writer()
    parallel._send_result(lambda: 42, writer)
    assert writer.sent == [(True, 42)] and writer.closed


def test_a_background_part_that_fails_sends_its_traceback():
    def fail():
        raise KeyError("missing page")

    writer = _Writer()
    parallel._send_result(fail, writer)
    ((ok, text),) = writer.sent
    assert ok is False and writer.closed
    assert "Traceback" in text and "KeyError: 'missing page'" in text


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs fork")
def test_a_forked_background_part_that_fails_raises_in_the_parent():
    def fail():
        raise ValueError("bad tree")

    result = parallel.in_background(fail, jobs=2)
    with pytest.raises(RuntimeError, match="a background part of the build failed") as error:
        result()
    assert "ValueError: bad tree" in str(error.value)
