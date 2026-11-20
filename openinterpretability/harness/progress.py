"""Progress lines with a rate and the time left, for steps that can run for hours.

A scan or an attribution on the CPU can take hours for a model above about 1B parameters
(issue #3971). Without output, a slow run looks the same as a stopped one. A
:class:`Progress` prints one line to stderr at most once every :data:`INTERVAL` seconds,
so a fast run prints nothing new.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

INTERVAL = float(os.environ.get("OPENINTERPRETABILITY_PROGRESS_SECONDS", "60"))
"""Seconds between two progress lines (``$OPENINTERPRETABILITY_PROGRESS_SECONDS``)."""


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


@dataclass
class Progress:
    """Counts the items of one step, and says how far it is at most once an interval."""

    what: str  # for example "pieces scanned"
    total: int
    interval: float = INTERVAL
    clock: object = time.monotonic
    done: int = 0
    _start: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._start = self._last = self.clock()

    def line(self) -> str:
        """``what: done/total, elapsed, about N left`` for the count so far."""
        elapsed = self.clock() - self._start
        text = f"{self.what}: {self.done}/{self.total}, {_duration(elapsed)} so far"
        if self.done and self.total > self.done:
            left = elapsed / self.done * (self.total - self.done)
            text += f", about {_duration(left)} left"
        return text

    def step(self, n: int = 1) -> str | None:
        """Count ``n`` more items. Print and return a line if an interval has passed."""
        self.done += n
        now = self.clock()
        if now - self._last < self.interval:
            return None
        self._last = now
        text = self.line()
        print(text, file=sys.stderr, flush=True)
        return text
