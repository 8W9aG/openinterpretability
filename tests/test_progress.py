"""Progress lines for long steps (openinterp/harness/progress.py, issue #3971)."""

from openinterp.harness.progress import Progress


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_a_line_at_most_once_an_interval_with_the_time_left(capsys):
    clock = Clock()
    progress = Progress("examples attributed", 5, interval=60, clock=clock)
    clock.now = 30
    assert progress.step() is None  # under a minute: nothing new
    clock.now = 120
    line = progress.step()
    assert line == "examples attributed: 2/5, 2 min so far, about 3 min left"
    assert line in capsys.readouterr().err
    clock.now = 150
    assert progress.step() is None


def test_long_runs_are_given_in_hours():
    clock = Clock()
    progress = Progress("pieces", 10, interval=0, clock=clock)
    clock.now = 3 * 3600
    assert progress.step(5) == "pieces: 5/10, 3.0 h so far, about 3.0 h left"
