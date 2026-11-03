"""Measure the WordNet specificity check (:class:`~openinterp.core.specificity.WordnetSpecificity`).

Each word goes into every carrier sentence in ``CARRIERS``. Its strength in one carrier
is the unit's peak over the tokens that overlap the word's characters. Its recorded
strength is the mean over the carriers. No carrier starts with the word, so a unit with
a first-token floor still has the word's tokens measured.
"""

from __future__ import annotations

from collections.abc import Sequence

from openinterp import wordnet
from openinterp.core.schemas import ACTIVATION_THRESHOLD
from openinterp.core.specificity import WordnetSpecificity, WordStrength
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.units import Unit

CARRIERS = (
    "Here is a word: {}.",
    "They talked about {} for a while.",
    "The next entry in the list is {}.",
)
MAX_WORDS = 40  # words on each side, most frequent first
MIN_WORDS = 5  # fewer on either side and there is nothing to compare


def word_strength(runner: NeuronHookRunner, unit: Unit, carrier: str, word: str) -> float | None:
    """Return the unit's peak over ``word``'s tokens in ``carrier``.

    ``None`` when no measured token overlaps the word.
    """
    text = carrier.format(word)
    start = carrier.index("{}")
    end = start + len(word)
    offsets = runner.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    strengths = unit.strengths(runner, text)[0]
    if len(offsets) != strengths.numel():
        return None
    first = int(unit.skip_first)
    held = [
        float(strengths[i])
        for i, (a, b) in enumerate(offsets)
        if i >= first and a < end and b > start
    ]
    return max(held) if held else None


def mean_strength(runner: NeuronHookRunner, unit: Unit, word: str) -> float | None:
    """The mean of :func:`word_strength` over ``CARRIERS``, or ``None`` if none measures it."""
    found = [s for c in CARRIERS if (s := word_strength(runner, unit, c, word)) is not None]
    return sum(found) / len(found) if found else None


def measure(runner: NeuronHookRunner, unit: Unit, words: Sequence[str]) -> list[WordStrength]:
    """Each word's strength, rounded; words that no carrier measures are left out."""
    out = []
    for word in words:
        s = mean_strength(runner, unit, word)
        if s is not None:
            out.append(WordStrength(word=word, strength=round(max(0.0, s), 6)))
    return out


def auroc(category: Sequence[float], siblings: Sequence[float]) -> float:
    """The chance that a category value beats a sibling value, ties counting half."""
    wins = sum((a > b) + 0.5 * (a == b) for a in category for b in siblings)
    return wins / (len(category) * len(siblings))


def specificity(
    runner: NeuronHookRunner,
    unit: Unit,
    lex: wordnet.Lexicon,
    synset: str,
    source: str,
    scale: float,
    exclude: Sequence[str] = (),
) -> WordnetSpecificity | None:
    """Return the unit on ``synset``'s words against its siblings' words.

    Words in ``exclude`` (the scan's peak words) are left out of both lists. ``None``
    when either list has fewer than ``MIN_WORDS`` measured words.
    """
    skip = {w.lower() for w in exclude}
    inside = wordnet.holdouts(lex, synset, skip, MAX_WORDS)
    near, outside = wordnet.sibling_words(
        lex, synset, skip | {w.lower() for w in inside}, MAX_WORDS
    )
    category, siblings = measure(runner, unit, inside), measure(runner, unit, outside)
    if len(category) < MIN_WORDS or len(siblings) < MIN_WORDS:
        return None
    line = ACTIVATION_THRESHOLD * scale
    a, b = [w.strength for w in category], [w.strength for w in siblings]
    return WordnetSpecificity(
        synset=synset,
        lemma=lex.lemma(synset),
        source=source,
        sibling_synsets=near,
        carriers=list(CARRIERS),
        scale=scale,
        category=category,
        siblings=siblings,
        category_active=round(sum(x >= line for x in a) / len(a), 6),
        siblings_active=round(sum(x >= line for x in b) / len(b), 6),
        auroc=round(auroc(a, b), 6),
    )
