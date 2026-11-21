"""Takedown lines and the spans that their shingles cover (core/takedowns.py)."""

from openinterpretability.core import takedowns as td


def test_blank_lines_between_takedowns_are_skipped():
    line = '{"id": "a-1", "date": "2026-11-01", "reason": "asked", "shingles": ["00"]}'
    takedowns = td.parse_takedowns(f"{line}\n\n   \n{line.replace('a-1', 'b-2')}\n")
    assert [t.id for t in takedowns] == ["a-1", "b-2"]


def test_overlapping_shingles_merge_into_one_span():
    passage = "one two three four five six seven eight nine"  # nine words: two shingles
    hashes = frozenset(td.shingle_hashes(passage))
    assert len(hashes) == 2
    text = f"Before. {passage} after."
    start = text.index("one")
    assert td.covered(text, hashes) == ((start, start + len(passage)),)
    assert td.redact(text, hashes, "[gone]") == "Before. [gone] after."
