"""The language mix of an experiment's scan, which its page shows beside its corpus.

A scan of a corpus whose documents carry ``lang`` counts its tokens per language
(``ConfoundProfile.labels``). So a record knows the language mix of what it scanned.
Each experiment scans its own seeded sample, so its page shows its own mix (#3298).
Tokens are the model's: one corpus splits into more tokens of some languages under one
tokenizer than under another.
"""

from __future__ import annotations

from dataclasses import dataclass

from openinterpretability.core.confounds import LANGUAGE

SHOWN_LANGUAGES = 8  # languages listed per mix; the rest are summed as "other"


@dataclass(frozen=True)
class LanguageShare:
    """One language's tokens in a scan."""

    label: str
    tokens: int
    share: float  # of the scan's labelled tokens


@dataclass(frozen=True)
class LanguageMix:
    """The languages of one scan's labelled tokens, most frequent first."""

    model_name: str
    tokens: int  # labelled tokens
    sampled: bool  # from a --max-tokens sample of the corpus, not the whole of it
    shares: list[LanguageShare]

    @property
    def shown(self) -> list[LanguageShare]:
        """The languages that the page lists by name."""
        return self.shares[:SHOWN_LANGUAGES]

    @property
    def other(self) -> float:
        """The share of labelled tokens in languages not :attr:`shown`.

        The record lists at most 50 languages, so this includes any that it left out.
        """
        return max(0.0, 1.0 - sum(s.share for s in self.shown))


def language_mix(submission) -> LanguageMix | None:
    """The language mix of the submission's scan, or ``None`` if it had no language labels."""
    confounds = submission.confounds
    found = (confounds.labels or {}).get(LANGUAGE) if confounds is not None else None
    if found is None or not found.labelled_tokens:
        return None
    total = found.labelled_tokens
    shares = [LanguageShare(s.label, s.count, s.count / total) for s in found.levels]
    sampled = submission.provenance is not None and submission.provenance.corpus_sample is not None
    return LanguageMix(submission.identifier.model_name, total, sampled, shares)
