"""Fail if any source or documentation file exceeds the line limit.

Tests, configuration, lockfiles and generated data (audit records and wiki
pages) are exempt. Run via `make lint`; CI runs it in the lint step.
"""

from __future__ import annotations

import subprocess
import sys
from fnmatch import fnmatch

MAX_LINES = 500

EXEMPT = (
    "tests/*",
    "wiki/*",  # generated experiment pages
    "docs/analyses/*.jsonl",  # generated analysis rows
    "hf_models/*.jsonl",  # dictionary training logs
    "uv.lock",
    "*.toml",
    "*.yml",
    "*.yaml",
    ".gitignore",
    ".python-version",
    "LICENSE",
)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [f for f in out.splitlines() if f and not any(fnmatch(f, p) for p in EXEMPT)]


def main() -> int:
    too_long = []
    for name in tracked_files():
        try:
            with open(name, encoding="utf-8") as fh:
                count = sum(1 for _ in fh)
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary or deleted files
        if count > MAX_LINES:
            too_long.append((count, name))
    for count, name in sorted(too_long, reverse=True):
        print(f"{name}: {count} lines (max {MAX_LINES}); split it into smaller modules")
    return 1 if too_long else 0


if __name__ == "__main__":
    sys.exit(main())
