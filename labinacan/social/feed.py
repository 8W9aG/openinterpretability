"""Read the entries of an Atom feed (issue #4463)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ATOM = "{http://www.w3.org/2005/Atom}"
THREAD = "{https://open-interpretability.com/ns/social}thread"  # research_feeds.THREAD_NS


@dataclass(frozen=True)
class Media:
    """The drawn head figure that an entry links as an enclosure (#5534)."""

    url: str
    type: str  # image/png or video/mp4
    alt: str  # the figure's alt text, for the post's media


@dataclass(frozen=True)
class Tweet:
    """One tweet of an entry's thread: its text and its media, at most 4 images (#5833)."""

    text: str
    media: tuple[Media, ...] = ()


@dataclass(frozen=True)
class Entry:
    """One entry: its stable id (a URL), title, link, summary, head figure and thread."""

    id: str
    title: str
    link: str
    summary: str
    media: Media | None = None
    thread: tuple[Tweet, ...] = ()  # empty: the entry posts as one post


def _thread(element) -> tuple[Tweet, ...]:
    """The thread that an entry's ``<oi:thread>`` holds as JSON, or () when it has none.

    Args:
        element: The ``<oi:thread>`` element, or None.

    Returns:
        The tweets, in order.

    Examples:
        >>> import xml.etree.ElementTree as ET
        >>> found = ET.fromstring('<t>[{"text": "Hi", "media": []}]</t>')
        >>> _thread(found), _thread(None)
        ((Tweet(text='Hi', media=()),), ())
    """
    import json

    if element is None or not (element.text or "").strip():
        return ()
    return tuple(Tweet(t["text"], tuple(Media(m["url"], m["type"], m["alt"]) for m in t["media"]))
                 for t in json.loads(element.text))  # fmt: skip


def read_feed(path: Path) -> list[Entry]:
    """The entries of the Atom feed at ``path``, in the feed's order (newest first).

    Args:
        path: The Atom feed's file.

    Returns:
        One entry per ``<entry>`` element, in the feed's order.

    Examples:
        >>> import tempfile
        >>> feed = Path(tempfile.mkdtemp()) / "feed.xml"
        >>> _ = feed.write_text(
        ...     '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>u1</id>'
        ...     '<title>T</title><link href="https://x.org/a"/><summary>S</summary>'
        ...     '</entry></feed>')
        >>> read_feed(feed)
        [Entry(id='u1', title='T', link='https://x.org/a', summary='S', media=None, thread=())]
        >>> _ = feed.write_text(
        ...     '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>u1</id><title>T</title>'
        ...     '<link href="https://x.org/a"/><link rel="enclosure" href="https://x.org/m.png" '
        ...     'type="image/png" title="A curve"/></entry></feed>')
        >>> read_feed(feed)[0].media
        Media(url='https://x.org/m.png', type='image/png', alt='A curve')
    """
    # The feeds are files that this repo's own build wrote, not data from outside.
    root = ET.parse(path).getroot()  # noqa: S314
    out = []
    for e in root.findall(f"{ATOM}entry"):
        links = e.findall(f"{ATOM}link")
        link = next((x for x in links if x.get("rel", "alternate") == "alternate"), None)
        enclosure = next((x for x in links if x.get("rel") == "enclosure"), None)
        media = (
            None
            if enclosure is None
            else Media(
                enclosure.get("href", ""), enclosure.get("type", ""), enclosure.get("title", "")
            )
        )
        out.append(Entry(id=(e.findtext(f"{ATOM}id") or "").strip(),
                         title=(e.findtext(f"{ATOM}title") or "").strip(),
                         link=link.get("href", "") if link is not None else "",
                         summary=(e.findtext(f"{ATOM}summary") or "").strip(),
                         media=media, thread=_thread(e.find(THREAD))))  # fmt: skip
    return out
