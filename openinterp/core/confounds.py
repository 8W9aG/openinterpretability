"""Schemas for confound profiles.

A confound profile says how much trivial explanations (token identity, position,
per-token labels such as language) account for a unit's activation.

Recorded in every scan, never gated (docs/PROTOCOL.md §4). The harness that measures
them is ``openinterp.harness.confounds`` and, for labels, ``openinterp.harness.annotations``.
"""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel, UnitFloat

# The label set of a document's language (a corpus's ``lang`` field; ``corpus.py``).
LANGUAGE = "language"
# The label set of a blended corpus's document's preset (its ``source`` field).
SOURCE = "source"
# The label set of a code document's programming language (a corpus's ``code_language``).
PROGRAMMING_LANGUAGE = "programming language"
# The label sets of a code document's syntax tree (`openinterp/syntax.py`).
SYNTAX_NODE = "syntax node"
BRACKET_DEPTH = "bracket depth"
ROUND_DEPTH = "round depth"
INDENT_DEPTH = "indent depth"
SYNTAX_CONSTRUCT = "syntax construct"
# The label sets of what `sensitive.py`, `toxicity.py` and `names.py` find in a document,
# and the label of every other token. Tokens with no label leave R² and mutual
# information, so a label set that named only the matches would always measure zero.
PERSONAL_DATA = "personal data"
TOXICITY = "toxicity"  # a model's guess (`openinterp/toxicity.py`)
HATE_SPEECH = "hate speech"  # a model's guess (`openinterp/toxicity.py`)
NAMES = "names"  # a model's guess (`openinterp/names.py`)
POSTAL_ADDRESSES = "postal addresses"
NO_MATCH = "none"
# Each word's universal part-of-speech tag, from the names model (`openinterp/names.py`).
# It labels every word, so it is not a match set: its whitespace gaps are NO_MATCH.
PART_OF_SPEECH = "part of speech"
MATCH_LABEL_SETS = (PERSONAL_DATA, TOXICITY, HATE_SPEECH, NAMES, POSTAL_ADDRESSES)
# Recorded, but not in the proposer's prompt. Most units have an active token in some name,
# so every prompt would carry this noisy label.
NOT_SHOWN_TO_PROPOSER = (NAMES,)


class TokenStat(StrictModel):
    """One token's mean strength over the scan, for the strongest-tokens table."""

    token: str
    token_id: int = Field(ge=0)
    mean_strength: float = Field(ge=0.0)
    count: int = Field(gt=0)


class LabelStat(StrictModel):
    """One label's count, mean strength and active tokens over the labelled tokens."""

    label: str
    count: int = Field(gt=0)
    mean_strength: float = Field(ge=0.0)
    active: int = Field(ge=0, description="Tokens at or above the protocol's activation line.")


class LabelConfound(StrictModel):
    """How much one label set explains of the pole's activation.

    A label set is, for example, a token's language or its part of speech. The fields
    are ``r2``, as ``token_r2`` is for token identity, and the binned,
    bias-corrected mutual information between strength and label, in bits. Both are
    over labelled tokens only. ``levels`` lists the most frequent labels.
    """

    r2: UnitFloat
    mutual_information_bits: float = Field(ge=0.0)
    labelled_tokens: int = Field(ge=0)
    unlabelled_tokens: int = Field(ge=0)
    levels: list[LabelStat]

    @property
    def has_active_match(self) -> bool:
        """Return whether a match label set has active tokens; most units have none."""
        return any(s.label != NO_MATCH and s.active > 0 for s in self.levels)


class ConfoundProfile(StrictModel):
    """How much of the pole's activation simple confounds explain over the scan corpus.

    ``token_r2`` / ``position_r2`` are the variance explained by a per-token-id / per-
    position mean model. ``trigger_tokens`` (highest mean strength, ``count >= min_count``)
    and ``position_means`` drive the trivial baselines a hypothesis must match or beat.
    """

    token_r2: UnitFloat
    position_r2: UnitFloat
    frequency_correlation: float = Field(ge=-1.0, le=1.0)
    min_count: int = Field(gt=0)
    trigger_tokens: list[TokenStat]
    position_means: list[float]
    labels: dict[str, LabelConfound] | None = Field(
        default=None,
        description="Confounds from per-token labels (e.g. 'language'), by label set; "
        "None when the scan had none. Reported, not gated.",
    )
