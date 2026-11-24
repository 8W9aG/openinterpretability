"""The literature map's input from GitHub and its report (#5798, #6529).

``scripts/literature_map.py`` places each queued paper (an open `area: literature` issue)
next to the reviews, and reports what it wrote. These parts moved here from that script,
which reached its line limit (#6529).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "8W9aG/openinterpretability"
MIN_COMPACT = 0.8  # the least compactness the map may have (tests/test_literature_map.py)


# Words that every queued issue shares, which would pull the queue into one cluster (#5798).
BOILERPLATE = re.compile(
    r"\*\*(Off-topic|Bears on):\*\*|Write a short not-applicable page\.", re.IGNORECASE
)


def plain_lead(text: str) -> str:
    """``text`` without the issue template's markers and stock sentence.

    Args:
        text: The issue's lead paragraph.

    Returns:
        The paragraph's own words.

    Examples:
        >>> plain_lead("**Off-topic:** World models. Write a short not-applicable page.")
        'World models.'
    """
    return " ".join(BOILERPLATE.sub("", text).split())


def queued() -> list[dict]:
    """The open `area: literature` issues: number, title and the text to embed."""
    try:
        out = subprocess.run(
            ["gh", "issue", "list", "--repo", REPO, "--label", "area: literature", "--state",
             "open", "--limit", "500", "--json", "number,title,body"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for issue in json.loads(out):
        title = re.sub(r"^Review\s+", "", issue["title"]).strip()
        lead = (issue["body"] or "").split("\n\n")[1:2]  # the paragraph after the source line
        rows.append({"number": issue["number"], "title": title,
                     "text": f"{title}\n\n{plain_lead(' '.join(lead))}".strip()})  # fmt: skip
    return sorted(rows, key=lambda r: r["number"])


def report(data: dict, out: Path) -> None:
    """Print what the map holds, and warn when its regions are not compact enough.

    Args:
        data: The map, as written.
        out: Where it was written.
    """
    print(
        f"{len(data['reviews'])} reviews, {len(data['queued'])} queued, "
        f"{len(data['regions'])} regions, compactness {data['compactness']} "
        f"({data['layout']}) → {out}"
    )
    if data["compactness"] < MIN_COMPACT:
        print(
            f"Compactness is under {MIN_COMPACT}, so tests/test_literature_map.py fails. "
            "Look at the layout before you commit it.",
            file=sys.stderr,
        )
