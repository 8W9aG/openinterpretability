"""Labelled concept sets from annotated Hugging Face datasets (`openinterp concepts`).

A concept set is JSON lines ``{"text": ..., "spans": [[start, end], ...]}``, the labelled
text that ``openinterp probe-compare`` reads: character spans where the concept holds,
marked by human annotators, never by the model. It feeds sparse-probing searches and
gives reviewers positive holdouts from a source the proposer never saw. A
``<name>.source.json`` sidecar (:class:`~openinterp.core.probing.ConceptSource`) pins
the dataset commit, split, label and licence.

Two kinds of label are read:

* *word tags* (CoNLL-2003 entities and Penn Treebank tags, Universal Dependencies parts
  of speech and relations): a span over each word with the tag. A tag matches a label
  exactly, without its BIO prefix (``B-PER``, ``I-PER`` -> ``PER``) or without its
  subtype (``nmod:poss`` -> ``nmod``). Sentences are joined back into documents.
* *document classes* (language, news topic): a span over the whole text of each
  document of the class. The sample is balanced: half the documents are of the class,
  half are of the others.

Splits are small (a few MB), so each is read whole, then sampled with a seed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from openinterp.core.probing import ConceptSource
from openinterp.corpus import _license, source_path
from openinterp.harness.annotations import Span


@dataclass(frozen=True)
class ConceptPreset:
    dataset: str
    files: str  # glob within the dataset repo, with {split} and, for treebanks, {subset}
    split: str  # the default split: a held-out one, which is small
    column: str  # word tags, or one class per row
    # How rows become documents: "conll" (pre-tokenised words, joined with spaces), "ud"
    # (words located in each sentence's text) or "document" (one per row, one class).
    layout: str
    revision: str | None = None
    subset: str | None = None  # default treebank, for "ud"
    bio: bool = False  # tags carry B-/I- prefixes
    text_column: str = "text"


# Licences differ, and some forbid redistribution: check before quoting excerpts in the wiki.
CONCEPT_PRESETS = {
    # CoNLL-2003 English NER (PER, ORG, LOC, MISC) over Reuters news of 1996. Licence
    # "other": the text is Reuters RCV1, for research use, so don't publish excerpts. The
    # main branch is a loading script, so read the Hub's Parquet conversion.
    "conll2003-ner": ConceptPreset(
        "eriktks/conll2003",
        "conll2003/{split}/*.parquet",
        "validation",
        "ner_tags",
        "conll",
        revision="refs/convert/parquet",
        bio=True,
    ),
    # The same sentences' Penn Treebank tags (NNP, VBD, ...): a second part-of-speech
    # source, independent of Universal Dependencies' annotators and genres.
    "conll2003-pos": ConceptPreset(
        "eriktks/conll2003",
        "conll2003/{split}/*.parquet",
        "validation",
        "pos_tags",
        "conll",
        revision="refs/convert/parquet",
    ),
    # Universal Dependencies parts of speech (NOUN, VERB, ADP, ...) for any of its ~300
    # treebanks; --subset picks one (en_ewt: English web text; en_gum: eight genres).
    # Licences are per treebank (en_ewt CC BY-SA 4.0, en_gum CC BY-NC-SA 4.0).
    "ud-upos": ConceptPreset(
        "universal-dependencies/universal_dependencies",
        "parquet/{subset}/{split}.parquet",
        "dev",
        "upos",
        "ud",
        subset="en_ewt",
    ),
    # The same treebanks' dependency relations (nsubj, obj, amod, ...), for syntax neurons.
    "ud-deprel": ConceptPreset(
        "universal-dependencies/universal_dependencies",
        "parquet/{subset}/{split}.parquet",
        "dev",
        "deprel",
        "ud",
        subset="en_ewt",
    ),
    # Sentences and reviews in 20 languages (ar, bg, de, el, en, es, fr, hi, it, ja, nl,
    # pl, pt, ru, sw, th, tr, ur, vi, zh). No licence on the card; its sources include
    # Amazon reviews and XNLI (CC BY-NC 4.0). The main branch is CSV.
    "language-identification": ConceptPreset(
        "papluca/language-identification",
        "default/{split}/*.parquet",
        "validation",
        "labels",
        "document",
        revision="refs/convert/parquet",
    ),
    # AG News headlines and lead sentences in four topics (World, Sports, Business,
    # Sci/Tech). Licence "unknown": academic, non-commercial use.
    "ag_news": ConceptPreset(
        "fancyzhx/ag_news", "data/{split}-*.parquet", "test", "label", "document"
    ),
}


@dataclass(frozen=True)
class Sentence:
    text: str
    spans: list[Span]  # one per tagged word
    newdoc: bool = False
    newpar: bool = True


# Pre-tokenised CoNLL text: "lobster ," and "( AP )" are joined as written, "lobster,", "(AP)".
_ATTACH_LEFT = re.compile(r"[.,;:!?%)\]}]+|n't|'(?:s|re|ve|m|ll|d)", re.IGNORECASE)
_ATTACH_RIGHT = re.compile(r"[(\[{$]")


def join_words(words: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Pre-tokenised words as text, with each word's characters [start, end)."""
    text, offsets = "", []
    for i, word in enumerate(words):
        if i and not (_ATTACH_LEFT.fullmatch(word) or _ATTACH_RIGHT.fullmatch(words[i - 1])):
            text += " "
        offsets.append((len(text), len(text) + len(word)))
        text += word
    return text, offsets


def locate_words(
    text: str, words: Sequence[str], mwt: Sequence[dict] = ()
) -> list[tuple[int, int]] | None:
    """Each word's characters in its sentence's ``text``. The words of a multiword token
    (French "du" = "de" + "le") share its surface form's characters. ``None`` if a word
    isn't where it should be, which would misplace every label after it."""
    fused = {}
    for token in mwt or ():
        first, last = token["id"].split("-")
        fused[int(first) - 1] = (int(last) - 1, token["form"])
    out: list[tuple[int, int]] = []
    cursor = i = 0
    while i < len(words):
        last, form = fused.get(i, (i, words[i]))
        start = text.find(form, cursor) if form else -1
        if start < 0 or text[cursor:start].strip():
            return None
        cursor = start + len(form)
        out += [(start, cursor)] * (last - i + 1)
        i = last + 1
    return out


def group_sentences(sentences: Sequence[Sentence], max_chars: int) -> list[tuple[str, list[Span]]]:
    """Consecutive sentences joined into documents of at most ``max_chars``: a new one at
    each new source document or when the next sentence wouldn't fit. Paragraphs are
    separated by newlines, sentences within one by a space. A sentence longer than
    ``max_chars`` is cut, and its spans with it."""
    docs: list[tuple[str, list[Span]]] = []
    text, spans = "", []
    for sent in sentences:
        sep = "\n" if sent.newpar else " "
        if text and (sent.newdoc or len(text) + len(sep) + len(sent.text) > max_chars):
            docs.append((text, spans))
            text, spans = "", []
        shift = len(text) + len(sep) if text else 0
        text = (text + sep if text else "") + sent.text
        spans += [Span(s.start + shift, s.end + shift, s.label) for s in sent.spans]
        if len(text) > max_chars:
            text = text[:max_chars]
            spans = [
                Span(s.start, min(s.end, max_chars), s.label) for s in spans if s.start < max_chars
            ]
    if text:
        docs.append((text, spans))
    return docs


def tag_matches(tag: str, label: str) -> bool:
    return tag == label or tag.split(":")[0] == label


def _tag(tag, names: list[str] | None, bio: bool) -> str | None:
    """A word's or row's label as text, or ``None`` for none (``O``, ``_``)."""
    if tag is None:
        return None
    tag = names[tag] if names is not None and isinstance(tag, int) else str(tag)
    if tag in ("O", "_", ""):
        return None
    return re.sub(r"^[BI]-", "", tag) if bio else tag


def class_names(schema, column: str) -> list[str] | None:
    """The names of a Hugging Face ClassLabel column (or a sequence of them), stored as
    integers in Parquet, from the file's ``huggingface`` schema metadata."""
    meta = (schema.metadata or {}).get(b"huggingface")
    if meta is None:
        return None
    feature = json.loads(meta).get("info", {}).get("features", {}).get(column, {})
    feature = feature.get("feature", feature)  # Sequence(ClassLabel) holds it inside
    return feature.get("names")


def _read_rows(fs, paths: Sequence[str], columns: Sequence[str]) -> tuple[list[dict], list | None]:
    rows, names = [], None
    for path in sorted(paths):
        with fs.open(path, "rb") as fh:
            file = pq.ParquetFile(fh)
            names = names or class_names(file.schema_arrow, columns[-1])
            rows += file.read(columns=list(columns)).to_pylist()
    return rows, names


def _documents(preset: ConceptPreset, column: str, rows, names, max_chars: int):
    """The split's documents, each with a span per tagged word or class."""
    if preset.layout == "document":
        docs = []
        for row in rows:
            text = (row[preset.text_column] or "").strip()[:max_chars]
            label = _tag(row[column], names, preset.bio)
            if text:
                docs.append((text, [Span(0, len(text), label)] if label else []))
        return docs, 0
    sentences, skipped = [], 0
    for row in rows:
        words = row["tokens"] or []
        if preset.layout == "conll":
            text, offsets = join_words(words)
            newdoc, newpar = False, True
        else:
            text = row[preset.text_column] or ""
            offsets = locate_words(text, words, row.get("mwt"))
            comments = row.get("comments") or []
            newdoc = any(c.startswith("newdoc") for c in comments)
            newpar = newdoc or any(c.startswith("newpar") for c in comments)
        if not words or offsets is None:
            skipped += 1
            continue
        spans = [
            Span(start, end, label)
            for (start, end), tag in zip(offsets, row[column], strict=True)
            if (label := _tag(tag, names, preset.bio)) is not None
        ]
        sentences.append(Sentence(text, spans, newdoc, newpar))
    return group_sentences(sentences, max_chars), skipped


@dataclass
class LoadedSplit:
    docs: list[tuple[str, list[Span]]]
    dataset: str
    revision: str
    files: str
    license: str | None
    skipped_sentences: int

    def counts(self) -> Counter:
        """How many words (or documents) carry each label."""
        return Counter(span.label for _, spans in self.docs for span in spans)


def load_split(
    name: str,
    *,
    split: str | None = None,
    subset: str | None = None,
    column: str | None = None,
    max_chars: int = 3000,
) -> LoadedSplit:
    """A preset's split at a pinned commit, as labelled documents."""
    from huggingface_hub import HfApi, HfFileSystem

    preset = CONCEPT_PRESETS[name]
    column = column or preset.column
    api = HfApi()
    info = api.dataset_info(preset.dataset, revision=preset.revision)
    fs = HfFileSystem()
    subset = subset or preset.subset
    files = preset.files.format(split=split or preset.split, subset=subset)
    paths = fs.glob(f"datasets/{preset.dataset}@{info.sha}/{files}")
    if not paths:
        raise FileNotFoundError(f"no files match {files!r} in {preset.dataset}@{info.sha}")
    if preset.layout == "document":
        columns = [preset.text_column, column]
    elif preset.layout == "conll":
        columns = ["tokens", column]
    else:
        columns = [preset.text_column, "tokens", "mwt", "comments", column]
    rows, names = _read_rows(fs, paths, columns)
    docs, skipped = _documents(preset, column, rows, names, max_chars)
    if preset.layout == "ud":
        # Universal Dependencies licenses each treebank separately.
        meta = json.loads(fs.read_text(f"datasets/{preset.dataset}@{info.sha}/metadata.json"))
        license_ = meta.get(subset, {}).get("license")
    else:
        license_ = _license(api, preset.dataset, info, preset.revision)
    return LoadedSplit(
        docs,
        preset.dataset,
        info.sha,
        files,
        ", ".join(license_) if isinstance(license_, list) else license_,
        skipped,
    )


def sample_concept(
    docs: Sequence[tuple[str, list[Span]]], label: str, documents: int, seed: int, balance: bool
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Up to ``documents`` documents with the spans of ``label``, in split order. With
    ``balance``, half (rounded up) have a span and half don't, where there are enough."""
    marked = [[(s.start, s.end) for s in spans if tag_matches(s.label, label)] for _, spans in docs]
    rng = random.Random(seed)
    if balance:
        pos = [i for i, m in enumerate(marked) if m]
        neg = [i for i, m in enumerate(marked) if not m]
        take_pos = min(len(pos), (documents + 1) // 2)
        picked = rng.sample(pos, take_pos) + rng.sample(neg, min(len(neg), documents - take_pos))
    else:
        picked = rng.sample(range(len(docs)), min(documents, len(docs)))
    return [(docs[i][0], marked[i]) for i in sorted(picked)]


def build_concept(
    name: str,
    label: str,
    out: Path,
    *,
    split: str | None = None,
    subset: str | None = None,
    column: str | None = None,
    documents: int = 1000,
    seed: int = 0,
    max_chars: int = 3000,
) -> ConceptSource:
    """Write ``label``'s concept set from preset ``name`` and its source sidecar."""
    preset = CONCEPT_PRESETS[name]
    loaded = load_split(name, split=split, subset=subset, column=column, max_chars=max_chars)
    counts = loaded.counts()
    if not any(tag_matches(tag, label) for tag in counts):
        known = ", ".join(tag for tag, _ in counts.most_common())
        raise ValueError(f"no {label!r} labels in {name}; it has: {known}")
    level = "document" if preset.layout == "document" else "token"
    rows = sample_concept(loaded.docs, label, documents, seed, balance=level == "document")
    data = "".join(
        json.dumps({"text": text, "spans": [list(s) for s in spans]}, ensure_ascii=False) + "\n"
        for text, spans in rows
    ).encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    source = ConceptSource(
        preset=name,
        dataset=loaded.dataset,
        revision=loaded.revision,
        files=loaded.files,
        column=column or preset.column,
        label=label,
        level=level,
        documents=len(rows),
        positive_documents=sum(1 for _, spans in rows if spans),
        spans=sum(len(spans) for _, spans in rows),
        skipped_sentences=loaded.skipped_sentences,
        seed=seed,
        max_chars=max_chars,
        license=loaded.license,
        sha256=hashlib.sha256(data).hexdigest(),
    )
    source_path(out).write_text(source.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return source
