"""Fail when the project's old name comes back (issue #4089).

The old name belongs to an unrelated project. It may appear only in the places below, each
for a reason that ``openinterpretability/core/legacy_names.py`` gives. Run by ``make lint``.
"""

from __future__ import annotations

import re
import subprocess
import sys

OLD = "open" + "interp"  # spelled in two parts, so this file doesn't match itself
PATTERN = re.compile(OLD + r"(?!retability)", re.IGNORECASE)
# Frozen: signed records and their signatures cover these, so they keep the old name.
# The X account's handle is a name that X holds, not this project's old name (#5389).
FROZEN = (OLD + "_version", OLD + "-record", "x.com/" + OLD + "ret")
ALLOWED_PATHS = (
    "wiki/",  # write-once, signed records
    "benchmark/v1/",  # the sealed benchmark
    ".github/allowed_signers",  # the signing namespace of every existing key
    "openinterpretability/core/legacy_names.py",  # the fallbacks for users' machines
    "tests/test_legacy_names.py",
    "CHANGELOG.md",  # the rename's entry lists the old variable names
    "changelog.d/4089.changed.md",
)


def offending(lines: list[str]) -> list[str]:
    """The ``path:line:text`` lines that use the old name outside the allowed places."""
    found = []
    for line in lines:
        path, _, rest = line.partition(":")
        if path.startswith(ALLOWED_PATHS):
            continue
        text = rest
        for name in FROZEN:
            text = text.replace(name, "")
        if PATTERN.search(text):
            found.append(line)
    return found


def main() -> int:
    """Grep the tracked files, and report each use of the old name."""
    out = subprocess.run(
        ["git", "grep", "-nIi", OLD, "--", ".", ":!wiki"],
        capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip
    found = offending(out.splitlines())
    for line in found:
        print(line[:200])
    if found:
        print(f"{len(found)} uses of the old name (issue #4089); use 'openinterpretability'.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
