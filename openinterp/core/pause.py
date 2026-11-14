"""A pause of new experiments on this machine, for a measurement that needs a quiet machine.

``open-interpretability pause`` writes a file, ``$OPENINTERP_PAUSE_FILE`` (default
``~/.cache/openinterp/pause``). While it is there, ``open-interpretability next`` draws no
experiment, and ``scripts/experiment-loop.sh`` starts no session. Experiments that
already run go on to the end. ``open-interpretability resume`` removes the file.

A pause ends by itself at its ``until`` time, so a pause that nobody removes can't stop
the loops for ever. A file that can't be read counts as a pause that started when the
file was last written, for :data:`DEFAULT_MINUTES`.
"""

from __future__ import annotations

import getpass
import json
import os
import secrets
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ENV = "OPENINTERP_PAUSE_FILE"
DEFAULT_MINUTES = 120
EXIT_PAUSED = 75  # EX_TEMPFAIL from sysexits.h: try again later


@dataclass(frozen=True)
class Pause:
    """Why new experiments wait, who asked for it, and until when."""

    reason: str
    by: str
    since: str  # ISO 8601, UTC
    until: str

    def describe(self) -> str:
        """One line that says why new experiments wait, and until when."""
        until = datetime.fromisoformat(self.until).astimezone().strftime("%H:%M")
        return (
            f"New experiments are paused until {until}: {self.reason} (by {self.by}). "
            "Run `open-interpretability resume` to continue sooner."
        )


def pause_file() -> Path:
    """``$OPENINTERP_PAUSE_FILE``, else ``$XDG_CACHE_HOME/openinterp/pause``."""
    if value := os.environ.get(ENV):
        return Path(value)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "openinterp" / "pause"


def read(now: datetime | None = None) -> Pause | None:
    """The pause in force, or None. It removes a pause whose time has ended."""
    path = pause_file()
    now = now or datetime.now(UTC)
    try:
        data = json.loads(path.read_text())
        pause = Pause(**{k: str(data[k]) for k in ("reason", "by", "since", "until")})
        ends = datetime.fromisoformat(pause.until)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        since = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        ends = since + timedelta(minutes=DEFAULT_MINUTES)
        pause = Pause("the pause file can't be read", "?", since.isoformat(), ends.isoformat())
    if ends <= now:
        with suppress(OSError):
            path.unlink()
        return None
    return pause


def start(reason: str, minutes: float = DEFAULT_MINUTES, now: datetime | None = None) -> Pause:
    """Pause new experiments for ``minutes``, and return the pause."""
    now = now or datetime.now(UTC)
    who = f"{getpass.getuser()}, process {os.getpid()}"
    until = now + timedelta(minutes=minutes)
    pause = Pause(reason, who, now.isoformat(), until.isoformat())
    path = pause_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    tmp.write_text(json.dumps(asdict(pause), indent=1))
    os.replace(tmp, path)
    return pause


def stop() -> bool:
    """End the pause. Return True if there was one."""
    try:
        pause_file().unlink()
    except FileNotFoundError:
        return False
    return True
