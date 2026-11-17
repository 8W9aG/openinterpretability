"""Write the taxonomy table of docs/PROTOCOL.md §1 from openinterpretability/core/taxonomy.py.

Run it after a change to the taxonomy. ``--check`` writes nothing, and fails if the table differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.core.taxonomy import protocol_table  # noqa: E402

PROTOCOL = Path(__file__).resolve().parents[1] / "docs" / "PROTOCOL.md"
START, END = "<!-- taxonomy:start -->\n", "<!-- taxonomy:end -->"


def updated(text: str) -> str:
    """``text`` with the table between the markers replaced by the taxonomy's."""
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    return before + START + protocol_table() + END + after


def main() -> int:
    """Write the table, or compare it with the file. Return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the table differs.")
    args = parser.parse_args()
    text = PROTOCOL.read_text(encoding="utf-8")
    new = updated(text)
    if args.check:
        if new != text:
            print("docs/PROTOCOL.md: the taxonomy table is stale; run scripts/taxonomy_table.py")
            return 1
        return 0
    PROTOCOL.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
