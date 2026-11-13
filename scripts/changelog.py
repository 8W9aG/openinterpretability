"""Changelog fragments: one file per change in changelog.d/, gathered at each release.

Every pull request used to add its entry at the top of CHANGELOG.md's Unreleased
section. So each merge made every other open pull request conflict. Now a change adds
its own file, ``changelog.d/<name>.<type>.md``. The name is the issue number, or any
short slug. The type is one of the Keep a Changelog headings in lower case: added,
changed, deprecated, removed, fixed or security. The file holds one or more bullets, in
the form CHANGELOG.md uses.

    python3 scripts/changelog.py check
    python3 scripts/changelog.py collect 0.0.20 2026-11-14 "Ships with protocol version **0.8**."

``check`` runs in ``make lint``. It fails on a badly named or empty fragment, and on an
entry in CHANGELOG.md's Unreleased section, which must stay empty. ``collect`` is the
release step. It writes the version's section from the fragments, then deletes them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
FRAGMENTS = ROOT / "changelog.d"
# Keep a Changelog's headings, in the order a version section lists them.
TYPES = ("added", "changed", "deprecated", "removed", "fixed", "security")
NAME = re.compile(rf"^[\w-]+\.({'|'.join(TYPES)})\.md$")
UNRELEASED = "## [Unreleased]\n"


def fragments(folder: Path = FRAGMENTS) -> list[Path]:
    """Every fragment file, sorted by name. README.md explains the folder and is skipped."""
    return sorted(p for p in folder.glob("*.md") if p.name != "README.md")


def problems(changelog: Path = CHANGELOG, folder: Path = FRAGMENTS) -> list[str]:
    """What ``check`` reports: bad fragment names, empty fragments, Unreleased entries."""
    out = []
    for path in fragments(folder):
        if not NAME.match(path.name):
            out.append(f"{path.name}: name it <issue or slug>.<{'|'.join(TYPES)}>.md")
        elif not path.read_text().lstrip().startswith("- "):
            out.append(f"{path.name}: start it with a '- ' bullet")
    text = changelog.read_text()
    start = text.index(UNRELEASED) + len(UNRELEASED)
    end = text.find("\n## [", start)
    if text[start : end if end >= 0 else None].strip():
        out.append(
            "CHANGELOG.md: the Unreleased section must be empty. "
            "Put the entry in changelog.d/ instead (scripts/changelog.py)."
        )
    return out


def section(version: str, date: str, intro: str, folder: Path = FRAGMENTS) -> str:
    """The version's CHANGELOG section, built from the fragments."""
    by_type: dict[str, list[str]] = {t: [] for t in TYPES}
    for path in fragments(folder):
        by_type[path.name.split(".")[-2]].append(path.read_text().strip())
    parts = [f"## [{version}] - {date}", ""]
    if intro:
        parts += [intro, ""]
    for kind in TYPES:
        if by_type[kind]:
            parts += [f"### {kind.capitalize()}", "", *by_type[kind], ""]
    return "\n".join(parts) + "\n"


def collect(version: str, date: str, intro: str = "", changelog: Path = CHANGELOG,
            folder: Path = FRAGMENTS) -> int:  # fmt: skip
    """Write the version's section under Unreleased, delete the fragments, return their count."""
    found = fragments(folder)
    text = changelog.read_text()
    at = text.index(UNRELEASED) + len(UNRELEASED)
    rest = text[at:].lstrip("\n")
    changelog.write_text(text[:at] + "\n" + section(version, date, intro, folder) + "\n" + rest)
    for path in found:
        path.unlink()
    return len(found)


def main() -> None:
    """``check``, or ``collect <version> <date> [intro]``."""
    args = sys.argv[1:]
    if args[:1] == ["check"]:
        found = problems()
        for problem in found:
            print(f"changelog: {problem}", file=sys.stderr)
        sys.exit(1 if found else 0)
    if args[:1] == ["collect"] and len(args) in (3, 4):
        count = collect(args[1], args[2], args[3] if len(args) == 4 else "")
        print(f"Collected {count} fragments into CHANGELOG.md [{args[1]}]")
        return
    sys.exit("usage: changelog.py check | collect <version> <YYYY-MM-DD> [intro]")


if __name__ == "__main__":
    main()
