"""Long loops report their progress (harness/progress.py, #3142)."""

from types import SimpleNamespace

from openinterpretability.builder import parallel
from openinterpretability.harness import circuit_measures as cm
from openinterpretability.harness import progress


class Recorder:
    """A Progress that records its counts instead of printing."""

    made: list = []

    def __init__(self, what, total):
        self.what, self.total, self.done = what, total, 0
        Recorder.made.append(self)

    def step(self, n=1):
        self.done += n


def test_a_circuit_review_counts_its_random_sets(monkeypatch):
    Recorder.made = []
    monkeypatch.setattr(progress, "Progress", Recorder)
    monkeypatch.setattr(cm, "random_like", lambda runner, members, rng: members)
    monkeypatch.setattr(cm, "measures", lambda *a: {"necessity": 0.1})
    assert cm.null_necessity(None, None, None, [], sets=7) == [0.1] * 7
    (made,) = Recorder.made
    assert (made.what, made.total, made.done) == ("random member sets", 7, 7)


def test_unit_pages_count_in_one_process(monkeypatch):
    Recorder.made = []
    monkeypatch.setattr(parallel, "Progress", Recorder)
    seen = []
    render = SimpleNamespace(jobs=1)
    parallel.render_in_parallel(render, lambda x: seen.append(x), [(1,), (2,), (3,)])
    (made,) = Recorder.made
    assert seen == [1, 2, 3] and (made.what, made.total, made.done) == ("unit pages", 3, 3)


def test_a_line_names_the_count_and_the_time_left():
    ticks = iter(range(0, 1000, 30))  # each read of the clock is 30 s later
    p = progress.Progress("things", 4, interval=0.0, clock=lambda: float(next(ticks)))
    p.step()
    assert p.step() == "things: 2/4, 2 min so far, about 2 min left"
