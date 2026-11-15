"""Remove personal data, and passages under a takedown, from the text that the site shows.

Records keep their text unchanged, because experiments study this text, and records are
signed. Only the build's output changes. :mod:`openinterp.sensitive` does the search.

The search is most of a build's time, and records never change. So the matches of each
text are kept on disk, in ``$XDG_CACHE_HOME/openinterp/pii`` (``$OPENINTERP_PII_CACHE``
moves it, and ``off`` turns it off). The file's name holds a hash of the search's code and
package versions, so a change to either starts a new cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from openinterp import sensitive
from openinterp.core import takedowns
from openinterp.core.cache_dirs import cache_root
from openinterp.core.schemas import AuditRecord
from openinterp.sensitive import pii_matches

REMOVED_TEXT = "[removed]"  # replaces a match in free text
REMOVED_TOKEN = "█"  # noqa: S105  (a model token, not a password). Replaces each token of a match, so a heatmap keeps its shape

Span = tuple[int, int]


_found: dict[str, tuple[Span, ...]] = {}  # searched in this run, not yet on disk


@cache
def _cache_file() -> Path | None:
    """Return the cache file for this search's code and package versions, or ``None``."""
    root = cache_root("OPENINTERP_PII_CACHE", "pii")
    if root is None:
        return None
    code = Path(sensitive.__file__).read_bytes() + Path(__file__).read_bytes()
    key = hashlib.sha256(sensitive.detector().encode() + code).hexdigest()[:16]
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


@cache
def find_spans(text: str) -> tuple[Span, ...]:
    """Return the ``(start, end)`` of each match in ``text``, sorted and merged.

    A match is personal data, or a passage that a takedown names
    (:mod:`openinterp.core.takedowns`).
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
        new = {key: scrub(item) for key, item in value.items()}
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


def _scrub_attribution(submission):
    """Remove attribution tokens that a match touched in the top example they come from.

    Attribution stores loose tokens, so the search cannot read them as one text.
    """
    attribution = submission.input_attribution
    if attribution is None:
        return submission

    def at(context, offset: int, token: str) -> str:
        if context.example >= len(submission.top_examples):
            return token
        example = submission.top_examples[context.example]
        position = example.max_token_index + offset
        removed = 0 <= position < len(example.tokens) and example.tokens[position] == REMOVED_TOKEN
        return REMOVED_TOKEN if removed else token

    examples = [
        context.model_copy(
            update={
                "peak_token": at(context, 0, context.peak_token),
                "top": [
                    t.model_copy(update={"token": at(context, t.offset, t.token)})
                    for t in context.top
                ],
            }
        )
        for context in attribution.examples
    ]
    return submission.model_copy(
        update={"input_attribution": attribution.model_copy(update={"examples": examples})}
    )


def scrub_record(record: AuditRecord) -> AuditRecord:
    """Return a copy of ``record`` to publish, with personal data removed from all its text.

    The copy reaches every page and the Parquet export, including fields added later. Call
    this after computing anything from probe text. The record on disk does not change.
    """
    scrubbed = scrub(record)
    return scrubbed.model_copy(update={"submission": _scrub_attribution(scrubbed.submission)})
