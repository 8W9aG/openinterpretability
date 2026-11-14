"""Add a takedown for a passage, and remove the passage from the experiment pages.

A maintainer runs this for a request to remove text (``SECURITY.md``, *A request to remove
text*). It reads the passage from a file that stays out of the repository. It hashes the
passage into shingles (:mod:`openinterp.core.takedowns`), and lists the records and pages
whose text the shingles match. It appends the takedown to ``wiki/takedowns.jsonl``. Then it
removes the matched text from each page, above its Discussion section.

    uv run python scripts/takedown.py --passage ~/request.txt --id 2026-11-14-a \\
        --reason "author's request"

Then sign the file (``open-interpretability sign wiki/takedowns.jsonl``), and open a pull request
with the file, its signature and the pages. Records and their signatures don't change.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from openinterp.builder.pii import REMOVED_TEXT
from openinterp.checks import DISCUSSION_HEADING
from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir
from openinterp.core.takedowns import (
    Takedown,
    all_hashes,
    covered,
    load_takedowns,
    redact,
    shingle_hashes,
    takedowns_path,
    to_line,
)


def strings(value) -> list[str]:
    """Every string in a record's JSON value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    if isinstance(value, dict):
        return [s for item in value.values() for s in strings(item)]
    return []


def matches(wiki_dir: Path, hashes: frozenset[str]) -> tuple[list[Path], list[Path]]:
    """The records, and the pages, whose text the shingles in ``hashes`` match."""
    root = experiments_dir(wiki_dir)
    records = [
        path
        for path in sorted(root.rglob("*.json"))
        if any(covered(s, hashes) for s in strings(json.loads(path.read_text(encoding="utf-8"))))
    ]
    pages = [
        path
        for path in sorted(root.rglob("*.md"))
        if path.name != "README.md" and covered(_head(path), hashes)
    ]
    return records, pages


def _head(page: Path) -> str:
    return page.read_text(encoding="utf-8").split(DISCUSSION_HEADING)[0]


def redact_page(page: Path, hashes: frozenset[str]) -> None:
    """Remove what ``hashes`` cover from ``page``, above its Discussion section."""
    text = page.read_text(encoding="utf-8")
    head, sep, rest = text.partition(DISCUSSION_HEADING)
    page.write_text(redact(head, hashes, REMOVED_TEXT) + sep + rest, encoding="utf-8")


def main() -> int:
    """Append the takedown, redact the pages, and say what to do next."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--passage", type=Path, required=True, help="A file with the passage.")
    parser.add_argument("--id", required=True, help="A new id, e.g. 2026-11-14-a.")
    parser.add_argument("--reason", required=True, help="A few words, with no personal data.")
    parser.add_argument("--wiki-dir", type=Path, default=DEFAULT_WIKI_DIR)
    opts = parser.parse_args()
    shingles = shingle_hashes(opts.passage.read_text(encoding="utf-8"))
    if opts.id in {t.id for t in load_takedowns(opts.wiki_dir)}:
        parser.error(f"takedown id {opts.id!r} is used already")
    hashes = frozenset(shingles)
    records, pages = matches(opts.wiki_dir, hashes)
    print(f"The passage matches {len(records)} record(s) and {len(pages)} page(s).")
    if not records and not pages:
        print("Nothing to take down. Check the passage against the site's text.")
        return 1
    takedown = Takedown(
        id=opts.id, date=datetime.now(UTC).date(), reason=opts.reason, shingles=shingles
    )
    path = takedowns_path(opts.wiki_dir)
    with path.open("a", encoding="utf-8") as f:
        f.write(to_line(takedown))
    # With every takedown's shingles, as `open-interpretability check` compares: a page added
    # after an earlier takedown can still hold that takedown's text.
    every = all_hashes(load_takedowns(opts.wiki_dir))
    for page in pages:
        redact_page(page, every)
    print(f"Appended {takedown.id} to {path}, and removed the passage from {len(pages)} page(s).")
    print(
        f"Next: open-interpretability sign {path}, then open a pull request with the file, its .sig"
    )
    print("and the pages. Rebuild and deploy the site after it merges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
