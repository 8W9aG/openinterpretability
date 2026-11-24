"""Remove personal data, and passages under a takedown, from the text that a site shows.

A project's records keep their text unchanged. Only the build's output changes.
:mod:`labinacan.sensitive` does the search, and :func:`scrub` walks any value's text.

The search is most of a build's time, and records never change. So the matches of each
text can be kept on disk, in a folder that the project gives to :func:`configure`. The
file's name holds a hash of the search's code, its name and its package versions. So a
change to any of them starts a new cache. With no folder, nothing is kept on disk.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from labinacan import sensitive
from labinacan.sensitive import pii_matches
from labinacan.site import takedowns

REMOVED_TEXT = "[removed]"  # replaces a match in free text
REMOVED_TOKEN = "█"  # noqa: S105  (a model token, not a password). Replaces each token of a match, so a heatmap keeps its shape

Span = tuple[int, int]


_found: dict[str, tuple[Span, ...]] = {}  # searched in this run, not yet on disk


@dataclass
class _Settings:
    """What the project gives the redaction core (:func:`configure`)."""

    cache_dir: Callable[[], Path | None] = lambda: None
    name: str = "labinacan.sensitive"


_settings = _Settings()


def configure(cache_dir: Callable[[], Path | None], name: str) -> None:
    """Set the folder that keeps the matches on disk, and the search's name.

    The folder is a function, read on the first search, so that a setting read from the
    environment can change before then. With no call, nothing is kept on disk, and the
    redaction is the same.

    Args:
        cache_dir: Returns the folder, or ``None`` to keep nothing on disk.
        name: The project's name for the search (:func:`labinacan.sensitive.detector`).

    Examples:
        >>> before = (_settings.cache_dir, _settings.name)  # the settings are global
        >>> configure(lambda: None, "lab.sensitive")
        >>> _cache_file() is None
        True
        >>> configure(*before)
    """
    _settings.cache_dir, _settings.name = cache_dir, name
    _cache_file.cache_clear()
    _stored.cache_clear()


@cache
def _cache_file() -> Path | None:
    """Return the cache file for this search's code and package versions, or ``None``."""
    root = _settings.cache_dir()
    if root is None:
        return None
    code = Path(sensitive.__file__).read_bytes() + Path(__file__).read_bytes()
    key = hashlib.sha256(sensitive.detector(_settings.name).encode() + code).hexdigest()[:16]
    return root / f"spans-{key}.sqlite"


@cache
def _stored() -> dict[str, tuple[Span, ...]]:
    """Return the matches on disk, by the hash of their text."""
    path = _cache_file()
    if path is None or not path.exists():
        return {}
    with sqlite3.connect(path, timeout=30) as db:
        rows = db.execute("SELECT text_hash, spans FROM spans").fetchall()
    return {h: tuple(tuple(span) for span in json.loads(spans)) for h, spans in rows}


def save_spans() -> None:
    """Add the matches that this run searched for to the cache file."""
    path = _cache_file()
    if path is None or not _found:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(h, json.dumps(spans)) for h, spans in _found.items()]
    with sqlite3.connect(path, timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS spans (text_hash TEXT PRIMARY KEY, spans TEXT)")
        db.executemany("INSERT OR IGNORE INTO spans VALUES (?, ?)", rows)
    _found.clear()


# A bounded cache: it holds each text as its key, and a build reads every record's text,
# which held over 1 GiB unbounded (#4652). The disk cache keeps the costly search's results.
@lru_cache(maxsize=4096)
def find_spans(text: str) -> tuple[Span, ...]:
    """Return the ``(start, end)`` of each match in ``text``, sorted and merged.

    A match is personal data, or a passage that a takedown names
    (:mod:`labinacan.site.takedowns`).
    """
    return _merge((*_personal_spans(text), *takedowns.covered(text, takedowns.active_hashes())))


def _personal_spans(text: str) -> tuple[Span, ...]:
    """The personal-data matches in ``text``, from the cache when it has them."""
    if not sensitive.may_match(text):
        return ()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest in (stored := _stored()):
        return stored[digest]
    _found[digest] = spans = _search(text)
    return spans


def _merge(spans: Sequence[Span]) -> tuple[Span, ...]:
    """``spans`` sorted, with overlapping and touching spans joined."""
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _search(text: str) -> tuple[Span, ...]:
    """Return the merged spans of the matches in ``text``, without the cache."""
    merged: list[Span] = []
    for start, end, _ in pii_matches(text):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def redact_text(text: str) -> str:
    """Return ``text`` with :data:`REMOVED_TEXT` in place of each match."""
    for start, end in reversed(find_spans(text)):
        text = text[:start] + REMOVED_TEXT + text[end:]
    return text


def redact_tokens(tokens: Sequence[str]) -> list[str]:
    """Return ``tokens`` with :data:`REMOVED_TOKEN` in place of each token that a match touches.

    The search reads the tokens joined, because a match usually spans several. The result
    has one item for each token, so it stays aligned with the activations.
    """
    out = list(tokens)
    position = 0
    spans = find_spans("".join(tokens))
    if not spans:
        return out
    for i, token in enumerate(tokens):
        end = position + len(token)
        if any(position < span_end and start < end for start, span_end in spans):
            out[i] = REMOVED_TOKEN
        position = end
    return out


def scrub(value: Any) -> Any:
    """Return ``value`` with every string in it redacted, or ``value`` itself if none changes."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        if not value or isinstance(value[0], (int, float)):
            return value  # a record's lists hold one type, and numbers hold no text
        new = [scrub(item) for item in value]
        return new if any(a is not b for a, b in zip(new, value, strict=True)) else value
    if isinstance(value, dict):
        new = {key: scrub_field(key, item) for key, item in value.items()}
        return new if any(new[key] is not value[key] for key in value) else value
    if isinstance(value, BaseModel):
        updates = {}
        for name in type(value).model_fields:
            field = getattr(value, name)
            new = redact_tokens(field) if name == "tokens" and _is_tokens(field) else scrub(field)
            if new != field:
                updates[name] = new
        return value.model_copy(update=updates) if updates else value
    return value


def _is_tokens(field: Any) -> bool:
    return isinstance(field, list) and all(isinstance(token, str) for token in field)


def scrub_field(name: str, value: Any) -> Any:
    """Scrub the field ``name`` of a dict, or ``value`` itself if none of it changes.

    A list of strings named ``tokens`` is a token list, so the search reads it joined.

    Args:
        name: The field's name.
        value: The field's value.

    Returns:
        The scrubbed value.

    Examples:
        >>> scrub_field("tokens", ["a", "b"])
        ['a', 'b']
    """
    if name == "tokens" and _is_tokens(value):
        new = redact_tokens(value)
        return new if new != value else value
    return scrub(value)
