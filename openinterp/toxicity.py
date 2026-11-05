"""Find toxic text and hate speech in a document, for the scan's label sets.

Both labels are models' guesses, for experiments only. Nothing removes text because of
them. The site's spoiler guard uses a word list (``sensitive.is_offensive``), so that a
site build needs no model.

A model reads text, not tokens. So each sentence is read alone, and all of its tokens take
the sentence's label. A label therefore says where toxic text is, not which word made it so.

The two models do different work. We compared them on HateCheck (Röttger et al., 2021),
which neither trained on, at a score of at least 0.5:

- ``unitary/unbiased-toxic-roberta`` (Apache 2.0) gives ``toxicity``. It alone finds
  obscenity, insults, threats and sexual text. Its identity outputs (``female``, ``black``)
  say that a text mentions a group, so we leave them out. Built from them, a target
  category found 0% to 76% of hateful cases, and 0% for immigrants, trans people and
  disabled people.
- ``cardiffnlp/twitter-roberta-base-hate-multiclass-latest`` gives ``hate speech``. It
  found 40% of hateful cases, and called 9% of non-hateful cases hate. It names the right
  target in 13% to 62% of a group's cases, but in 1% for disabled people. Its model card
  states no licence. It trained on tweets. We leave out its ``other`` output, because that
  output called lines of Ruby code hate.

Both models are English models. Scoring runs on the CPU, at about 10 ms a sentence for each
model. Each document's result stays on disk. The models (about 500 MB each) are downloaded
once, pinned to a commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

THRESHOLD = 0.5  # the score at which a sentence takes a label
_BATCH = 16
_MAX_CHARS = 1000  # a longer "sentence" (code, a table) is read in pieces of this length
_MAX_TOKENS = 512  # what the models accept
# On our corpus, the models called one-word lines hate or toxic ("Fluker.", "F.eks.").
_MIN_WORDS = 2
# Up to a stop followed by a space, else up to the end of the line.
_SENTENCE = re.compile(r"\S(?:[^\n]*?[.!?](?=\s)|[^\n]*)")


@dataclass(frozen=True)
class _Classifier:
    repo: str
    revision: str  # a commit, so a later scan reads the same weights
    name: str  # part of the cache's path
    labels: tuple[str, ...]  # the outputs that can label a sentence, most specific first
    softmax: bool = False  # the outputs compete; else each output is its own yes or no
    fallback: str | None = None  # the label of a sentence with no other label


_TOXICITY = _Classifier(
    "unitary/unbiased-toxic-roberta",
    "36295dd80b422dc49f40052021430dae76241adc",
    "toxicity",
    ("threat", "identity_attack", "sexual_explicit", "obscene", "insult"),
    fallback="toxicity",
)
_HATE = _Classifier(
    "cardiffnlp/twitter-roberta-base-hate-multiclass-latest",
    "b9a303f920f8527ac4151e65953c04505fdf0587",
    "hate",
    ("racism", "sexism", "religion", "sexual_orientation", "disability"),
    softmax=True,
)
DETECTOR = "; ".join(f"{c.repo}@{c.revision[:12]}" for c in (_TOXICITY, _HATE))
DETECTOR = f"{DETECTOR} (score at least {THRESHOLD}, per sentence of {_MIN_WORDS}+ words)"


def sentences(text: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` of each sentence in ``text``, split to ``_MAX_CHARS``."""
    return [
        (start, min(start + _MAX_CHARS, match.end()))
        for match in _SENTENCE.finditer(text)
        for start in range(match.start(), match.end(), _MAX_CHARS)
    ]


def _label(classifier: _Classifier, scores: dict[str, float]) -> str | None:
    """The strongest of a sentence's labels, else the fallback when its score passes."""
    passed = [label for label in classifier.labels if scores[label] >= THRESHOLD]
    if passed:
        return max(passed, key=scores.__getitem__)
    fallback = classifier.fallback
    return fallback if fallback and scores[fallback] >= THRESHOLD else None


@cache
def _model(classifier: _Classifier):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(classifier.repo, revision=classifier.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier.repo, revision=classifier.revision
    )
    return tokenizer, model.eval()


def _score(classifier: _Classifier, texts: Sequence[str]) -> list[dict[str, float]]:
    """Return each text's score per output of the model."""
    import torch

    tokenizer, model = _model(classifier)
    outputs = [model.config.id2label[i] for i in range(model.config.num_labels)]
    found: list[dict[str, float]] = []
    for first in range(0, len(texts), _BATCH):
        batch = tokenizer(
            list(texts[first : first + _BATCH]),
            truncation=True,
            max_length=_MAX_TOKENS,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**batch).logits
        scores = logits.softmax(-1) if classifier.softmax else logits.sigmoid()
        found += [dict(zip(outputs, row, strict=True)) for row in scores.tolist()]
    return found


def _cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "openinterp"


def _matches(classifier: _Classifier, text: str) -> tuple[tuple[int, int, str], ...]:
    """Return ``(start, end, label)`` of each labelled sentence, kept on disk by the text."""
    key = hashlib.sha256(text.encode()).hexdigest()
    version = f"{classifier.revision[:12]}-{THRESHOLD}-{_MIN_WORDS}"
    path = _cache_dir() / "toxicity" / classifier.name / version / f"{key}.json"
    if path.exists():
        return tuple((start, end, label) for start, end, label in json.loads(path.read_text()))
    spans = [(a, b) for a, b in sentences(text) if len(text[a:b].split()) >= _MIN_WORDS]
    scores = _score(classifier, [text[start:end] for start, end in spans]) if spans else []
    labels = [_label(classifier, s) for s in scores]
    found = [(*span, label) for span, label in zip(spans, labels, strict=True) if label]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.part")  # scans can run side by side
    partial.write_text(json.dumps(found))
    partial.replace(path)
    return tuple(found)


def toxicity_matches(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return ``(start, end, label)`` of each toxic sentence in ``text``, sorted."""
    return _matches(_TOXICITY, text)


def hate_matches(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return ``(start, end, label)`` of each sentence of hate speech in ``text``, sorted."""
    return _matches(_HATE, text)
