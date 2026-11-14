"""Takedowns: passages that the project stops publishing on request (``SECURITY.md``).

A takedown is one line of ``wiki/takedowns.jsonl``. It names a passage by the hashes of
its *shingles*: each run of :data:`SHINGLE_WORDS` consecutive words, lowercased. The file
never holds the text, because that would publish the text again. One passage can appear
in many records and fields, and the hashes find it in each one.

The build removes the text that the shingles cover from everything it publishes
(:mod:`openinterp.builder.pii`). An experiment page may change outside its Discussion
section only by that removal (:func:`redact`). The record files and their signatures
don't change. So a takedown hides the text without breaking reproduction.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from datetime import date as Date
from pathlib import Path

from pydantic import Field, ValidationError

from openinterp.core.paths import DEFAULT_WIKI_DIR
from openinterp.core.strict import StrictModel

SHINGLE_WORDS = 8  # a run this long is rare enough to name one passage
WORD = re.compile(r"\w+")
HASH_CHARS = 16  # a 64-bit prefix of SHA-256

Span = tuple[int, int]


class Takedown(StrictModel):
    """One request to stop publishing a passage."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    date: Date
    reason: str = Field(
        min_length=1, description="Why, in a few words. It must not hold personal data."
    )
    shingles: list[str] = Field(
        min_length=1, description="The hash of each shingle of the passage (shingle_hashes)."
    )


def takedowns_path(wiki_dir: Path | str = DEFAULT_WIKI_DIR) -> Path:
    """The takedown file of ``wiki_dir``."""
    return Path(wiki_dir) / "takedowns.jsonl"


def parse_takedowns(text: str) -> list[Takedown]:
    """Every takedown in ``text``, in order. Raises ``ValueError`` naming a bad line."""
    out = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(Takedown.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"line {number}: {exc.errors()[0]['msg']}") from exc
    return out


def load_takedowns(wiki_dir: Path | str = DEFAULT_WIKI_DIR) -> list[Takedown]:
    """Every takedown of ``wiki_dir``; none if there is no takedown file."""
    path = takedowns_path(wiki_dir)
    return parse_takedowns(path.read_text(encoding="utf-8")) if path.exists() else []


def _shingles(text: str) -> Iterable[tuple[str, Span]]:
    """Each shingle of ``text``: its hash, and the characters from its first to last word."""
    words = list(WORD.finditer(text))
    for i in range(len(words) - SHINGLE_WORDS + 1):
        run = words[i : i + SHINGLE_WORDS]
        key = " ".join(w.group().lower() for w in run)
        digest = hashlib.sha256(key.encode()).hexdigest()[:HASH_CHARS]
        yield digest, (run[0].start(), run[-1].end())


def shingle_hashes(passage: str) -> list[str]:
    """The hashes that name ``passage``, in order and without repeats.

    Raises ``ValueError`` for a passage shorter than one shingle.
    """
    hashes = list(dict.fromkeys(h for h, _ in _shingles(passage)))
    if not hashes:
        raise ValueError(f"a passage needs at least {SHINGLE_WORDS} words")
    return hashes


def all_hashes(takedowns: Sequence[Takedown]) -> frozenset[str]:
    """Every shingle hash of ``takedowns``."""
    return frozenset(h for t in takedowns for h in t.shingles)


def covered(text: str, hashes: frozenset[str]) -> tuple[Span, ...]:
    """The spans of ``text`` that the shingles in ``hashes`` cover, sorted and merged."""
    if not hashes:
        return ()
    merged: list[Span] = []
    for digest, (start, end) in _shingles(text):
        if digest not in hashes:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def redact(text: str, hashes: frozenset[str], marker: str) -> str:
    """``text`` with ``marker`` in place of each span that ``hashes`` cover."""
    for start, end in reversed(covered(text, hashes)):
        text = text[:start] + marker + text[end:]
    return text


_active: frozenset[str] = frozenset()


def activate(takedowns: Sequence[Takedown]) -> None:
    """Make the build remove what ``takedowns`` name (:func:`active_hashes`)."""
    global _active
    _active = all_hashes(takedowns)


def active_hashes() -> frozenset[str]:
    """The shingle hashes that the build removes. :func:`activate` sets them."""
    return _active


def to_line(takedown: Takedown) -> str:
    """``takedown`` as one line of the takedown file, with its keys in a fixed order."""
    return json.dumps(takedown.model_dump(mode="json")) + "\n"
