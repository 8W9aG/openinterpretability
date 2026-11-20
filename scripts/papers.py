"""Each paper's status and citation file, checked in lint (issue #3150).

A paper is a **preprint** while work on it is left, and it says which issues hold it
open. It is **published** when no work is left. After that its text and results don't
change. Later corrections go in a ``## Addenda`` section at the end. Each paper has a
``CITATION.cff`` (CFF 1.2.0) whose ``version`` agrees with its status line. A published
paper's file holds the SHA-256 of its text above the addenda and of its results files.
A change to them then fails the check.

A preprint's line must name only open issues (#5395). ``check --closing`` takes the issues a
pull request closes, and fails if a line still names one. ``check --online`` asks GitHub for
the state of every issue on every line, and fails on a closed one. Plain ``check`` stays
offline, for ``make lint``.

    uv run python scripts/papers.py check
    uv run python scripts/papers.py check --closing 4250,5394   # CI on a PR (#5395)
    uv run python scripts/papers.py check --online              # the nightly run (#5395)
    uv run python scripts/papers.py new <slug> --work "#N (what is left)"
    uv run python scripts/papers.py publish <slug>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.builder.research import (  # noqa: E402
    _ENTRY,
    ADDENDA,
    STATUS,
    paper_status,
)

PAPERS = Path(__file__).resolve().parents[1] / "papers"
SITE = "https://open-interpretability.com/research/"
REPO = "https://github.com/8W9aG/openinterpretability"
PREPRINT = "preprint"
PUBLISHED = "1.0"


def entries(papers: Path = PAPERS) -> list[tuple[str, str]]:
    """Each listed paper's (date, slug), from ``papers/README.md``."""
    return [(d, s) for d, s in _ENTRY.findall((papers / "README.md").read_text(encoding="utf-8"))]


def title(text: str) -> str:
    """The paper's level-1 heading."""
    return next(line[2:].strip() for line in text.splitlines() if line.startswith("# "))


def frozen_hash(folder: Path) -> str:
    """SHA-256 of the text above the addenda and of each results file, in name order."""
    digest = hashlib.sha256()
    text = (folder / "README.md").read_text(encoding="utf-8").split(ADDENDA)[0].rstrip()
    digest.update(text.encode())  # trailing blank lines don't count, so an addendum can start
    for path in sorted(folder.glob("results*.jsonl")):
        digest.update(path.name.encode() + b"\n" + path.read_bytes())
    return "sha256:" + digest.hexdigest()


def citation(folder: Path, date: str, version: str, digest: str | None = None) -> dict:
    """The paper's ``CITATION.cff``, as a mapping."""
    cff = {
        "cff-version": "1.2.0",
        "message": "If you use this paper, please cite it as below.",
        "type": "article",
        "title": title((folder / "README.md").read_text(encoding="utf-8")),
        "authors": [{"name": "Open Interpretability"}],
        "date-released": date,
        "version": version,
        "url": f"{SITE}{folder.name}/index.html",
        "repository-code": REPO,
    }
    if digest:
        cff["identifiers"] = [{"type": "other", "value": digest,
                               "description": "SHA-256 of the published text above its "
                               "addenda, and of its results files"}]  # fmt: skip
    return cff


def write_citation(folder: Path, cff: dict) -> None:
    """Write ``CITATION.cff`` in ``folder``."""
    text = yaml.safe_dump(cff, sort_keys=False, allow_unicode=True, width=1000)
    (folder / "CITATION.cff").write_text(text, encoding="utf-8")


def problems(folder: Path, date: str) -> list[str]:
    """What is wrong with one paper's status and citation file, if anything."""
    text = (folder / "README.md").read_text(encoding="utf-8")
    status, path = paper_status(text), folder / "CITATION.cff"
    if status is None:
        return ["no status line under the abstract (**Status:** Preprint. Work left: #N …)"]
    if status.preprint and not status.issues:
        return ["a preprint must name the issues that hold it open"]
    if not path.exists():
        return ["no CITATION.cff (scripts/papers.py new)"]
    cff = yaml.safe_load(path.read_text(encoding="utf-8"))
    want = citation(folder, date, PREPRINT if status.preprint else PUBLISHED)
    out = [f"CITATION.cff {key} is {cff.get(key)!r}, not {want[key]!r}"
           for key in ("cff-version", "type", "title", "date-released", "version", "url")
           if cff.get(key) != want[key]]  # fmt: skip
    if not status.preprint:
        stored = [i.get("value") for i in cff.get("identifiers", []) if i.get("type") == "other"]
        if stored != [frozen_hash(folder)]:
            out.append("the published text or results changed: add an addendum instead")
    return out


def check(papers: Path = PAPERS) -> list[str]:
    """Every problem of every listed paper, each as ``<slug>: <problem>``."""
    return [f"{slug}: {p}" for date, slug in entries(papers) for p in problems(papers / slug, date)]


def work_left(papers: Path = PAPERS) -> dict[str, tuple[int, ...]]:
    """Each preprint's issues of work left, by slug. A bracketed issue is a citation."""
    out = {}
    for _, slug in entries(papers):
        status = paper_status((papers / slug / "README.md").read_text(encoding="utf-8"))
        if status is not None and status.preprint:
            out[slug] = status.issues
    return out


FIX = "remove it from the status line, or run scripts/papers.py publish {slug} if no work is left"


def named(issues: set[int], papers: Path = PAPERS) -> list[str]:
    """A problem for each status line that names one of ``issues`` as work left."""
    return [f"{slug}: #{n} is closed, or closes in this PR, but the line names it: "
            f"{FIX.format(slug=slug)}"
            for slug, left in work_left(papers).items() for n in left if n in issues]  # fmt: skip


def closed(numbers: set[int]) -> set[int]:
    """Which of ``numbers`` are closed on GitHub."""
    import json
    import subprocess

    out = set()
    for n in sorted(numbers):
        view = subprocess.run(["gh", "issue", "view", str(n), "--repo", REPO, "--json", "state"],
                              capture_output=True, text=True, check=True)  # fmt: skip
        if json.loads(view.stdout)["state"] == "CLOSED":
            out.add(n)
    return out


def new(slug: str, work: str, papers: Path = PAPERS) -> None:
    """Give a paper its preprint status line, under the abstract, and its citation file."""
    folder = papers / slug
    text = (folder / "README.md").read_text(encoding="utf-8")
    if paper_status(text) is None:
        head, _, rest = text.partition("\n\n")  # the title
        abstract, _, body = rest.partition("\n\n")
        line = f"**Status:** Preprint. Work left: {work}"
        (folder / "README.md").write_text(f"{head}\n\n{abstract}\n\n{line}\n\n{body}")
    date = dict((s, d) for d, s in entries(papers))[slug]
    write_citation(folder, citation(folder, date, PREPRINT))


def publish(slug: str, papers: Path = PAPERS, today: str | None = None) -> None:
    """Publish a paper: its status line says so, and its citation file holds the hash."""
    folder = papers / slug
    today = today or datetime.now(UTC).strftime("%Y-%m-%d")
    readme = folder / "README.md"
    line = f"**Status:** Published {today}. Changes since are in the addenda at the end."
    readme.write_text(STATUS.sub(line, readme.read_text(encoding="utf-8"), count=1))
    date = dict((s, d) for d, s in entries(papers))[slug]
    write_citation(folder, citation(folder, date, PUBLISHED, frozen_hash(folder)))


def main() -> int:
    """Run the command. Return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check_cmd = sub.add_parser("check", help="Fail on a paper with no status, or a changed one.")
    check_cmd.add_argument(
        "--closing",
        default="",
        help="Issues a PR closes, e.g. 4250,5394: fail if a line names one.",
    )
    check_cmd.add_argument("--online", action="store_true",
                           help="Ask GitHub, and fail if a line names a closed issue.")  # fmt: skip
    one = sub.add_parser("new", help="Give a paper its preprint status and citation file.")
    one.add_argument("slug")
    one.add_argument("--work", required=True, help='The issues still open, e.g. "#12 (x)".')
    sub.add_parser("publish", help="Publish a paper with no work left.").add_argument("slug")
    args = parser.parse_args()
    if args.command == "new":
        new(args.slug, args.work)
    elif args.command == "publish":
        publish(args.slug)
    else:
        found = check()
        closing = {int(n.strip().lstrip("#")) for n in args.closing.split(",") if n.strip()}
        if args.online:
            closing |= closed({n for left in work_left().values() for n in left})
        found += named(closing)
        print("\n".join(found) or "Every paper has its status and citation file.")
        return 1 if found else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
