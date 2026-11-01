"""The corpora the registry's experiments were scanned on, and each one's language mix.

A scan of a corpus whose documents carry ``lang`` counts its tokens per language
(``ConfoundProfile.labels``), so a record knows the language mix of what it scanned.
Registry statistics over ``selection = "random"`` draws depend on that mix, so the site
shows it for each corpus. Tokens are the model's, so the mix is per model: one corpus
splits into more tokens of some languages under one tokenizer than under another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from openinterp.core.confounds import LANGUAGE

SHOWN_LANGUAGES = 8  # languages listed per mix; the rest are summed as "other"


@dataclass(frozen=True)
class LanguageShare:
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
        return self.shares[:SHOWN_LANGUAGES]

    @property
    def other(self) -> float:
        """The share of labelled tokens in languages not :attr:`shown` (the record lists
        at most 50 languages, so this includes any it left out)."""
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


@dataclass
class CorpusView:
    sha256: str | None
    name: str  # the provenance's description of the corpus
    experiments: int = 0
    mixes: list[LanguageMix] = field(default_factory=list)  # one per model, by name


def corpora_view(experiments: Sequence) -> list[CorpusView]:
    """Every corpus some experiment's provenance names, most used first, with its language
    mix under each model that scanned it with labels. Each mix is from the scan with the
    most labelled tokens, preferring scans of the whole corpus to samples of it."""
    corpora: dict[str | None, CorpusView] = {}
    best: dict[tuple[str | None, str], LanguageMix] = {}
    for e in experiments:
        sub = e.record.submission
        p = sub.provenance
        if p is None:
            continue
        view = corpora.setdefault(p.corpus_sha256, CorpusView(p.corpus_sha256, p.corpus))
        view.experiments += 1
        mix = language_mix(sub)
        if mix is None:
            continue
        key = (p.corpus_sha256, mix.model_name)
        old = best.get(key)
        if old is None or (not mix.sampled, mix.tokens) > (not old.sampled, old.tokens):
            best[key] = mix
    for (sha, _), mix in sorted(best.items(), key=lambda kv: kv[0][1]):
        corpora[sha].mixes.append(mix)
    return sorted(corpora.values(), key=lambda c: (-c.experiments, c.name))
