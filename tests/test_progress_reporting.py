"""Long loops report their progress (labinacan/progress.py, #3142)."""

from types import SimpleNamespace

from labinacan import progress
from labinacan.site import parallel
from openinterpretability.harness import circuit_measures as cm


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


def test_random_sets_take_the_task_and_seed_0_by_default(monkeypatch):
    import random

    monkeypatch.setattr(progress, "Progress", Recorder)
    draws, tasks = [], []
    monkeypatch.setattr(cm, "random_like", lambda runner, members, rng: draws.append(rng.random()))
    monkeypatch.setattr(
        cm, "measures", lambda r, task, m, s: tasks.append(task) or {"necessity": 0.2}
    )
    assert cm.null_necessity("runner", "task", "means", [], sets=2) == [0.2, 0.2]
    expected = random.Random(0)
    assert draws == [expected.random(), expected.random()] and tasks == ["task", "task"]
