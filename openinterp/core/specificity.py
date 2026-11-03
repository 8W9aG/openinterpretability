"""Schemas for the WordNet specificity check: a category's words against its siblings'.

A "cities" unit might really be a "places" unit. To tell the two apart, the harness
measures the unit on two lists of words. The first is words that WordNet puts under the
claimed category (its hyponyms). The second is words from the category's siblings
(other hyponyms of its hypernyms). Each word sits in a few fixed carrier sentences.
Peak words of the scan are left out of both lists, so every word is held out. Reported,
not gated (docs/PROTOCOL.md §4). Neither agent sees it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openinterp.core.strict import StrictModel, UnitFloat


class WordStrength(StrictModel):
    """One word's strength: the mean over the carriers of its peak over the word's tokens."""

    word: str = Field(min_length=1)
    strength: float = Field(ge=0.0)


class WordnetSpecificity(StrictModel):
    """The unit on a WordNet category's words against the words of its sibling categories.

    A word is active at or above ``ACTIVATION_THRESHOLD × scale``. ``auroc`` is the
    chance that a category word is stronger than a sibling word, ties counting half. It
    is 1 when the unit picks out the category. It is 0.5 when the unit can't tell the
    category from its siblings.
    """

    synset: str = Field(description="The category, e.g. 'city.n.01'.")
    lemma: str
    source: Literal["proposer", "peak_words"] = Field(
        description="'proposer': the draft's wordnet_synset. 'peak_words': the proposer named "
        "none, so the largest category of the scan's peak words."
    )
    sibling_synsets: list[str]
    carriers: list[str] = Field(min_length=1, description="Templates; {} is the word.")
    scale: float = Field(gt=0.0, description="The unit's top activation in the scan.")
    category: list[WordStrength]
    siblings: list[WordStrength]
    category_active: UnitFloat
    siblings_active: UnitFloat
    auroc: UnitFloat
