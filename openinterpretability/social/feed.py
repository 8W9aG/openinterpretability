"""Read the entries of an Atom feed (issue #4463)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class Entry:
    """One entry: its stable id (a URL), title, link and summary."""

    id: str
    title: str
    link: str
    summary: str


def read_feed(path: Path) -> list[Entry]:
    """The entries of the Atom feed at ``path``, in the feed's order (newest first)."""
    # The feeds are files that this repo's own build wrote, not data from outside.
    root = ET.parse(path).getroot()  # noqa: S314
    out = []
    for e in root.findall(f"{ATOM}entry"):
        link = e.find(f"{ATOM}link")
        out.append(Entry(id=(e.findtext(f"{ATOM}id") or "").strip(),
                         title=(e.findtext(f"{ATOM}title") or "").strip(),
                         link=link.get("href", "") if link is not None else "",
                         summary=(e.findtext(f"{ATOM}summary") or "").strip()))  # fmt: skip
    return out
