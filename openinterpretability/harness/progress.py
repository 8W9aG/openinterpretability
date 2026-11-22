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
TTY_INTERVAL = 0.2  # seconds between two redraws of a bar on a terminal
BAR_WIDTH = 20
QUIET_ENV = "OPENINTERPRETABILITY_QUIET"  # set by `explain --quiet`: no bars, no stage lines


def quiet() -> bool:
    """Whether ``$OPENINTERPRETABILITY_QUIET`` turns progress off (#5830).

    Returns:
        True when the variable is set to anything but "" or "0".

    Examples:
        >>> os.environ[QUIET_ENV] = "1"
        >>> quiet()
        True
        >>> del os.environ[QUIET_ENV]
        >>> quiet()
        False
    """
    return os.environ.get(QUIET_ENV, "") not in ("", "0")


def bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    """A text bar for ``done`` of ``total``.

    Args:
        done: The items done.
        total: All the items.
        width: The bar's width in characters, inside its brackets.

    Returns:
        ``[####......]``, filled in proportion.

    Examples:
        >>> bar(5, 10, width=10)
        '[#####.....]'
        >>> bar(3, 0, width=4)
        '[####]'
    """
    filled = width if total <= 0 else min(width, round(width * done / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


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
    tty: bool | None = None  # None: draw a bar when stderr is a terminal (#5830)
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
        """Count ``n`` more items. Print and return a line if an interval has passed.

        On a terminal the line is a bar, redrawn in place every :data:`TTY_INTERVAL`
        seconds and at the end. Elsewhere it is a plain line every ``interval`` seconds, so
        a log or an agent sees it. :func:`quiet` turns both off.
        """
        self.done += n
        if quiet():
            return None
        now = self.clock()
        tty = sys.stderr.isatty() if self.tty is None else self.tty
        end = self.done >= self.total
        if now - self._last < (TTY_INTERVAL if tty else self.interval) and not (tty and end):
            return None
        self._last = now
        text = self.line()
        if tty:
            print(f"\r{bar(self.done, self.total)} {text}", end="\n" if end else "",
                  file=sys.stderr, flush=True)  # fmt: skip
        else:
            print(text, file=sys.stderr, flush=True)
        return text


@dataclass
class Stages:
    """The stages of one command, each said as ``[i/N] what`` when it starts (#5830)."""

    total: int
    done: int = 0

    def next(self, what: str) -> str:
        """Start the next stage, and say so on stderr unless :func:`quiet`.

        Args:
            what: The stage, for example "Measuring the coverage".

        Returns:
            The line.

        Examples:
            >>> stages = Stages(3)
            >>> stages.next("Reading the corpus")
            '[1/3] Reading the corpus'
        """
        self.done += 1
        text = f"[{self.done}/{self.total}] {what}"
        if not quiet():
            print(text, file=sys.stderr, flush=True)
        return text
