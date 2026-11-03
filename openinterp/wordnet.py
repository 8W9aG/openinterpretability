"""WordNet categories of a unit's top-activating words, for the proposer and the reviewer.

The peak word of an example is the word that holds its peak token. :func:`categorise`
looks up the noun senses of each peak word in WordNet (the first ``MAX_SENSES``, which
WordNet lists most frequent first). It then gives the most specific hypernyms that the
words share ("11 of 15 are body parts"), and the words that fall outside each one.
Instance hypernyms count, so "Paris" is under "city".

The proposer gets the categories as candidate concepts and their levels of generality
(:func:`format_for_proposer`). The reviewer gets probe material for the largest one
(:func:`format_for_reviewer`). That is other hyponyms that no peak word is (positive
holdouts), sibling categories (hard negatives), and other senses of the peak words (the
polysemy illusion). Both are reported, never gated (docs/PROTOCOL.md §4).

Limits: English nouns only; words, not subword tokens; poor coverage of names, slang,
code and recent terms. No word-sense disambiguation is done: a word counts under a
category when any of its first senses is there.

The data is WordNet 3.0 through NLTK (Apache 2.0; WordNet's licence is BSD-like). NLTK
downloads it on first use. Without it, the agents get no WordNet note.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Protocol

from openinterp.core.examples import ActivatingExample

MAX_SENSES = 3  # noun senses of a word that count, most frequent first
MIN_LENGTH = 3  # shorter words ("a", "is") match unrelated abbreviations
MIN_WORDS = 3  # words a category must cover
MIN_DEPTH = 3  # shallower synsets (entity, abstraction, object) say nothing
MAX_GROUPS = 4
MAX_LISTED = 8  # holdouts, siblings and other senses given to the reviewer
MAX_GLOSS = 80  # characters of a gloss given to the reviewer
WORD = re.compile(r"[^\W\d_]+")


class Lexicon(Protocol):
    """The parts of WordNet that this module reads. Synsets are named ``"city.n.01"``."""

    def senses(self, word: str) -> list[str]:
        """The noun synsets of ``word``, most frequent first."""
        ...

    def hypernyms(self, synset: str) -> list[str]:
        """The direct hypernyms of ``synset``, instance hypernyms included."""
        ...

    def hyponyms(self, synset: str) -> list[str]:
        """The direct hyponyms of ``synset``, instance hyponyms included."""
        ...

    def lemma(self, synset: str) -> str:
        """The first lemma of ``synset``, with spaces (``"body part"``)."""
        ...

    def depth(self, synset: str) -> int:
        """The length of the shortest hypernym path from ``synset`` to the root."""
        ...

    def gloss(self, synset: str) -> str:
        """The definition of ``synset``."""
        ...

    def count(self, synset: str) -> int:
        """How often the lemmas of ``synset`` occur in WordNet's tagged corpus."""
        ...


class WordNet:
    """:class:`Lexicon` over NLTK's WordNet."""

    def __init__(self, wn) -> None:
        self.wn = wn

    def senses(self, word: str) -> list[str]:
        """The noun synsets of ``word``, most frequent first."""
        return [s.name() for s in self.wn.synsets(word, pos=self.wn.NOUN)]

    def hypernyms(self, synset: str) -> list[str]:
        """The direct hypernyms of ``synset``, instance hypernyms included."""
        s = self.wn.synset(synset)
        return [h.name() for h in s.hypernyms() + s.instance_hypernyms()]

    def hyponyms(self, synset: str) -> list[str]:
        """The direct hyponyms of ``synset``, instance hyponyms included."""
        s = self.wn.synset(synset)
        return [h.name() for h in s.hyponyms() + s.instance_hyponyms()]

    def lemma(self, synset: str) -> str:
        """The first lemma of ``synset``, with spaces (``"body part"``)."""
        return self.wn.synset(synset).lemma_names()[0].replace("_", " ")

    def depth(self, synset: str) -> int:
        """The length of the shortest hypernym path from ``synset`` to the root."""
        return self.wn.synset(synset).min_depth()

    def gloss(self, synset: str) -> str:
        """The definition of ``synset``."""
        return self.wn.synset(synset).definition()

    def count(self, synset: str) -> int:
        """How often the lemmas of ``synset`` occur in WordNet's tagged corpus."""
        return sum(lemma.count() for lemma in self.wn.synset(synset).lemmas())


@cache
def lexicon() -> Lexicon | None:
    """WordNet, downloaded by NLTK on first use, or ``None`` if it can't be had."""
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None
    try:
        wn.ensure_loaded()
    except LookupError:
        if not nltk.download("wordnet", quiet=True):
            return None
        try:
            wn.ensure_loaded()
        except LookupError:
            return None
    return WordNet(wn)


@dataclass(frozen=True)
class Category:
    """A synset that at least ``MIN_WORDS`` peak words fall under."""

    synset: str
    lemma: str
    depth: int
    words: tuple[str, ...]  # peak words under it, in first-seen order
    outside: tuple[str, ...]  # peak words in WordNet but not under it


@dataclass(frozen=True)
class Categories:
    """The peak words and the categories they share."""

    known: tuple[str, ...]  # peak words with a noun sense in WordNet
    missing: tuple[str, ...]  # peak words without one
    groups: tuple[Category, ...]  # most words first


def peak_word(ex: ActivatingExample) -> str | None:
    """Return the word, lower-cased, that holds the example's peak token.

    The word is the run of letters across token edges that overlaps the peak token's
    characters. ``None`` when the peak token has no letters.
    """
    peak = ex.max_token_index
    lo, hi = max(0, peak - 8), min(len(ex.tokens), peak + 8)
    before = "".join(ex.tokens[lo:peak])
    text = before + "".join(ex.tokens[peak:hi])
    start, end = len(before), len(before) + len(ex.tokens[peak])
    for match in WORD.finditer(text):
        if match.start() < end and match.end() > start:
            return match.group().lower()
    return None


def peak_words(examples: Sequence[ActivatingExample]) -> list[str]:
    """The distinct peak words of ``examples`` of at least ``MIN_LENGTH`` letters, in order."""
    words = (peak_word(ex) for ex in examples)
    return list(dict.fromkeys(w for w in words if w and len(w) >= MIN_LENGTH))


def ancestors(lex: Lexicon, synset: str) -> set[str]:
    """``synset`` and all its hypernyms, transitively."""
    seen, todo = set(), [synset]
    while todo:
        s = todo.pop()
        if s not in seen:
            seen.add(s)
            todo.extend(lex.hypernyms(s))
    return seen


def categorise(words: Sequence[str], lex: Lexicon) -> Categories:
    """Return the most specific categories that at least ``MIN_WORDS`` of ``words`` share.

    Deeper synsets are taken first. A shallower synset is kept only when no deeper kept
    synset covers the same set of words. So each kept category is the lowest shared
    hypernym of its words. The ``MAX_GROUPS`` with the most words are returned.
    """
    under: dict[str, set[str]] = {}
    known, missing = [], []
    for word in words:
        senses = lex.senses(word)[:MAX_SENSES]
        (known if senses else missing).append(word)
        for synset in set().union(*(ancestors(lex, s) for s in senses)):
            under.setdefault(synset, set()).add(word)
    depth = {s: lex.depth(s) for s, w in under.items() if len(w) >= MIN_WORDS}
    kept: dict[frozenset[str], str] = {}
    for synset in sorted(depth, key=lambda s: (-depth[s], s)):
        covered = frozenset(under[synset])
        if depth[synset] >= MIN_DEPTH and covered not in kept:
            kept[covered] = synset
    ranked = sorted(kept.items(), key=lambda kv: (-len(kv[0]), -depth[kv[1]], kv[1]))
    groups = tuple(
        Category(
            synset=synset,
            lemma=lex.lemma(synset),
            depth=depth[synset],
            words=tuple(w for w in known if w in covered),
            outside=tuple(w for w in known if w not in covered),
        )
        for covered, synset in ranked[:MAX_GROUPS]
    )
    return Categories(known=tuple(known), missing=tuple(missing), groups=groups)


def format_for_proposer(examples: Sequence[ActivatingExample], lex: Lexicon | None) -> str:
    """The proposer's note on the WordNet categories of the peak words, or ``""``."""
    if lex is None:
        return ""
    found = categorise(peak_words(examples), lex)
    if not found.groups:
        return ""
    lines = [
        f"- {g.lemma} ({g.synset}, depth {g.depth}): {len(g.words)} of {len(found.known)} "
        f"({', '.join(g.words)})" + (f"; outside: {', '.join(g.outside)}" if g.outside else "")
        for g in found.groups
    ]
    missing = f"Not in WordNet as nouns: {', '.join(found.missing)}.\n" if found.missing else ""
    return (
        "\nWordNet categories that the peak words share (nouns, first "
        f"{MAX_SENSES} senses of each word, no sense disambiguation; deeper is more "
        "specific). A candidate concept, not evidence: check it against the snippets.\n"
        + "\n".join(lines)
        + "\n"
        + missing
    )


def holdouts(lex: Lexicon, category: Category, limit: int = MAX_LISTED) -> list[str]:
    """Return lemmas of hyponyms of ``category`` that are not peak words, most frequent first."""
    seen, todo, found = set(), list(lex.hyponyms(category.synset)), []
    while todo:
        s = todo.pop()
        if s not in seen:
            seen.add(s)
            found.append(s)
            todo.extend(lex.hyponyms(s))
    lemmas: dict[str, int] = {}
    for s in found:
        lemma = lex.lemma(s)
        if lemma.lower() not in category.words:
            lemmas[lemma] = max(lemmas.get(lemma, 0), lex.count(s))
    return sorted(lemmas, key=lambda w: (-lemmas[w], w))[:limit]


def siblings(lex: Lexicon, category: Category, limit: int = MAX_LISTED) -> list[str]:
    """The lemmas of the other hyponyms of ``category``'s hypernyms, most frequent first."""
    found = {
        s
        for parent in lex.hypernyms(category.synset)
        for s in lex.hyponyms(parent)
        if s != category.synset
    }
    return [lex.lemma(s) for s in sorted(found, key=lambda s: (-lex.count(s), s))][:limit]


def other_senses(
    lex: Lexicon, category: Category, limit: int = MAX_LISTED
) -> list[tuple[str, str]]:
    """Return ``(word, gloss)`` for peak words under ``category`` with a sense outside it.

    The gloss is that of the word's most frequent noun sense that is not under
    ``category``, cut to ``MAX_GLOSS`` characters.
    """
    out = []
    for word in category.words:
        for sense in lex.senses(word)[:MAX_SENSES]:
            if category.synset not in ancestors(lex, sense):
                gloss = lex.gloss(sense)
                cut = gloss if len(gloss) <= MAX_GLOSS else gloss[: MAX_GLOSS - 1] + "…"
                out.append((word, cut))
                break
    return out[:limit]


def format_for_reviewer(examples: Sequence[ActivatingExample], lex: Lexicon | None) -> str:
    """The reviewer's WordNet probe material for the peak words' largest category, or ``""``."""
    if lex is None:
        return ""
    found = categorise(peak_words(examples), lex)
    if not found.groups:
        return ""
    top = found.groups[0]
    lines = [
        f"WordNet puts {len(top.words)} of the {len(found.known)} peak words found in it "
        f"under '{top.lemma}' ({', '.join(top.words)}). If the hypothesis claims this "
        "category or one near it, WordNet suggests:"
    ]
    if more := holdouts(lex, top):
        lines.append(f"- other '{top.lemma}' words, for positive holdouts: {', '.join(more)}")
    if near := siblings(lex, top):
        lines.append(f"- sibling categories, for hard negatives: {', '.join(near)}")
    if senses := other_senses(lex, top):
        listed = "; ".join(f"{w} ({gloss})" for w, gloss in senses)
        lines.append(f"- other senses of the peak words, for same-word hard negatives: {listed}")
    return "\n".join(lines) + "\n"
