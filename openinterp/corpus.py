"""Build scan corpora from Hugging Face datasets (`openinterp corpus`).

A corpus is a JSONL file of ``{"text": ...}`` documents, plus a ``<name>.source.json``
sidecar (:class:`CorpusSource`) that pins where it came from. Sampling does not download
whole datasets. It reads Parquet row groups over HTTP. So a 500-document sample of
fineweb-edu's 10B-token sample costs a few dozen small reads. :mod:`openinterp.corpus_cache`
keeps what it reads.

Sampling is deterministic for a given dataset revision and seed. It visits row groups in
a seeded random order across all files, and takes a seeded handful of documents from
each. This spreads the corpus over sources and crawl dumps.

The registry's scan corpus is not such a sample. It is pile-10k's first 500 documents,
each cut to 3,000 characters, with non-ASCII characters escaped. Rebuild it
byte for byte with ``openinterp corpus --dataset pile-10k --first --ascii`` (README).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from openinterp.core.confounds import LANGUAGE, PROGRAMMING_LANGUAGE, SOURCE
from openinterp.core.schemas import CorpusSample, CorpusSource
from openinterp.corpus_cache import read_row_group, row_groups
from openinterp.corpus_layouts import (  # noqa: F401  (wikitext_articles is re-exported)
    LAYOUTS,
    chat_formatter,
    wikitext_articles,
)
from openinterp.corpus_presets import (  # noqa: F401
    FINEWEB2_LANGUAGES,
    PRESETS,
    UNSTATED_LICENSES,
    WIKIPEDIA_LANGUAGES,
    Preset,
)

# What `openinterp audit` scans by default (protocol 0.7). The seed is fixed, so every
# audit of a corpus reads the same documents, and their scales are comparable.
DEFAULT_SAMPLE_TOKENS = 200_000
DEFAULT_SAMPLE_SEED = 0


# Document fields a corpus may carry besides "text", and the label set each gives the scan
# (`harness/annotations.py`). Labels from the dataset, or detected where the source
# sidecar names a detector (`lang_detector`, `code_detector`).
DOCUMENT_LABELS = {
    "lang": LANGUAGE,
    "source": SOURCE,
    "code_language": PROGRAMMING_LANGUAGE,
}


def source_path(corpus_path: Path) -> Path:
    """``corpus.jsonl`` -> ``corpus.source.json``."""
    return corpus_path.with_name(corpus_path.name.split(".")[0] + ".source.json")


def sample_documents(
    fs,
    paths: list[str],
    *,
    text_column: str,
    documents: int,
    seed: int,
    max_chars: int,
    docs_per_row_group: int,
    layout: str | None = None,
    label_column: str | None = None,
    labels: list | None = None,
    chat=None,
) -> list[str]:
    """Return up to ``documents`` non-empty texts from Parquet files on ``fs``.

    Each text is cut to ``max_chars``. Comma-separated ``text_column`` names join into one
    text (:func:`_read_groups`). With ``chat``, a function, each row's conversation in
    ``text_column`` becomes the text that ``chat`` returns. With a ``layout``, each row
    group's rows are first joined into its documents. With a ``label_column``, each
    returned text's value in that column joins the list ``labels``.
    """
    rng = random.Random(seed)
    paths = sorted(paths)
    # One footer read per file: in parallel, since a dataset of 880 files takes minutes serially.
    with ThreadPoolExecutor(16) as pool:
        counts = list(pool.map(lambda path: row_groups(fs, path), paths))
    groups = [(path, i) for path, n in zip(paths, counts, strict=True) for i in range(n)]
    rng.shuffle(groups)
    # Few row groups (a small dataset): take more from each, so the sample still fills.
    take = max(docs_per_row_group, math.ceil(documents / max(len(groups), 1)))
    out: list[str] = []
    for texts, column in _read_groups(fs, groups, text_column, label_column, chat):
        if layout is not None:
            texts = LAYOUTS[layout](texts)
        order = rng.sample(range(len(texts)), len(texts))
        picked = [j for j in order if (texts[j] or "").strip()][:take]  # empties don't count
        for j in sorted(picked):
            out.append(texts[j].strip()[:max_chars])
            if label_column:
                labels.append(column[j])
            if len(out) >= documents:
                return out
    return out


def first_documents(
    fs,
    paths: list[str],
    *,
    text_column: str,
    documents: int,
    max_chars: int,
    layout: str | None = None,
    label_column: str | None = None,
    labels: list | None = None,
    chat=None,
) -> list[str]:
    """Return the first ``documents`` non-empty texts of the Parquet files on ``fs``.

    The files are in sorted order. Each text is cut to ``max_chars`` but not stripped, as
    the registry's corpus was. ``label_column``, ``labels`` and ``chat`` are as in
    :func:`sample_documents`.
    """
    out: list[str] = []
    for path in sorted(paths):
        groups = [(path, i) for i in range(row_groups(fs, path))]
        for texts, column in _read_groups(fs, groups, text_column, label_column, chat):
            if layout is not None:
                texts = LAYOUTS[layout](texts)
            for j, text in enumerate(texts):
                if (text or "").strip():
                    out.append(text[:max_chars])
                    if label_column:
                        labels.append(column[j])
                    if len(out) >= documents:
                        return out
    return out


def _read_groups(
    fs, groups, text_column: str, label_column: str | None = None, chat=None
) -> Iterator[tuple[list, list | None]]:
    """Yield each row group's texts and, if there is a ``label_column``, that column.

    Comma-separated ``text_column`` names join into one text per row, with a blank line
    between them. A null value adds nothing, so a row with only nulls gives an empty text.
    With ``chat``, ``text_column`` is one column of conversations, and ``chat`` writes each
    as its row's text. An empty conversation gives an empty text.
    """
    text_columns = text_column.split(",")
    if chat and len(text_columns) > 1:
        raise ValueError("a chat template writes one column of conversations")
    columns = [*text_columns, label_column] if label_column else text_columns
    for path, i in groups:
        table = read_row_group(fs, path, i, columns)
        parts = [table.column(name).to_pylist() for name in text_columns]
        texts = (
            [chat(row) if row else "" for row in parts[0]]
            if chat
            else ["\n\n".join(filter(None, row)) for row in zip(*parts, strict=True)]
        )
        yield texts, table.column(label_column).to_pylist() if label_column else None


def jsonl_bytes(
    texts: Sequence[str], ascii: bool = False, labels: dict[str, Sequence[str]] | None = None
) -> bytes:
    """Return ``texts`` as a JSONL corpus, one ``{"text": ...}`` per line.

    Each of ``labels`` follows the text. ``labels`` maps a field of
    :data:`DOCUMENT_LABELS` to one value per document. With ``ascii``, non-ASCII
    characters are \\u-escaped.
    """
    labels = labels or {}
    rows = ({"text": t} | {k: v[i] for k, v in labels.items()} for i, t in enumerate(texts))
    return "".join(json.dumps(row, ensure_ascii=ascii) + "\n" for row in rows).encode()


def sample_tokens(
    tokenizer, texts: Sequence[str], max_tokens: int, seed: int
) -> tuple[list[str], CorpusSample]:
    """Return a seeded sample of whole documents, in corpus order, and its description.

    The sample has at most ``max_tokens`` tokens (see :class:`CorpusSample`).
    """
    picked, sample = sample_indices(tokenizer, texts, max_tokens, seed)
    return [texts[i] for i in picked], sample


def sample_indices(
    tokenizer, texts: Sequence[str], max_tokens: int, seed: int
) -> tuple[list[int], CorpusSample]:
    """:func:`sample_tokens`, as the sampled documents' indices in corpus order."""
    if not texts:
        raise ValueError("the corpus has no documents to sample")
    order = random.Random(seed).sample(range(len(texts)), len(texts))
    picked: list[int] = []
    total = 0
    for i in order:
        n = len(tokenizer.encode(texts[i], add_special_tokens=False))
        if picked and total + n > max_tokens:
            continue  # a shorter document later in the order may still fit
        picked.append(i)
        total += n
    picked.sort()
    sample = [texts[i] for i in picked]
    return picked, CorpusSample(
        seed=seed,
        max_tokens=max_tokens,
        documents=len(sample),
        corpus_documents=len(texts),
        tokens=total,
        sha256=hashlib.sha256(jsonl_bytes(sample)).hexdigest(),
    )


def write_corpus(
    texts: list[str], out: Path, *, ascii: bool = False, labels=None, **source
) -> CorpusSource:
    """Write the JSONL corpus, with each document's ``labels``, and its source sidecar.

    Returns the source.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    data = jsonl_bytes(texts, ascii, labels)
    out.write_bytes(data)
    record = CorpusSource(
        documents=len(texts), sha256=hashlib.sha256(data).hexdigest(), ascii=ascii, **source
    )
    source_path(out).write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record


def load_source(corpus_path: Path, sha256: str) -> CorpusSource | None:
    """The corpus's source sidecar, if it exists and describes these exact bytes."""
    path = source_path(corpus_path)
    if not path.exists():
        return None
    source = CorpusSource.model_validate_json(path.read_text(encoding="utf-8"))
    return source if source.sha256 == sha256 else None


def load_sources(paths: Sequence[Path]) -> dict[str, CorpusSource]:
    """Committed sources by the SHA-256 they describe; missing paths are skipped.

    A file holds one sidecar, or a list of sources (``corpus-sources.json``). The build
    uses them to show the source of records that scanned a corpus before its sidecar
    existed. It matches them by ``provenance.corpus_sha256``, and shows the source as
    inferred, not recorded.
    """
    sources = {}
    for path in paths:
        if Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else [data]:
                source = CorpusSource.model_validate(item)
                sources[source.sha256] = source
    return sources


def build_from_hub(
    dataset: str,
    out: Path,
    *,
    files: str,
    text_column: str = "text",
    revision: str | None = None,
    documents: int = 500,
    seed: int = 0,
    max_chars: int = 3000,
    docs_per_row_group: int = 20,
    layout: str | None = None,
    first: bool = False,
    ascii: bool = False,
    lang_pattern: str | None = None,
    detect_lang: bool = False,
    code_language_column: str | None = None,
    detect_code: bool = False,
    chat_template: str | None = None,
    chat_template_revision: str | None = None,
) -> CorpusSource:
    """Sample a corpus from a Hugging Face dataset at a pinned revision.

    With ``first``, take its first documents. With ``lang_pattern`` (see :class:`Preset`),
    each document records its glob's language as ``lang``. With ``detect_lang``, ``lang``
    is the language that :mod:`openinterp.langid` detects, where it is sure of one. With
    ``code_language_column``, each document records that column's value as ``code_language``.
    With ``detect_code``, ``code_language`` is the programming language that
    :mod:`openinterp.codeid` detects, where it is sure of one.
    With ``chat_template``, a model ID, ``text_column`` holds conversations. That model's
    chat template writes each one as a document. The template comes from the model's
    commit ``chat_template_revision`` (default: its latest commit, pinned on use).
    """
    texts, labels, source = _sample_hub(
        dataset,
        files=files,
        text_column=text_column,
        revision=revision,
        documents=documents,
        seed=seed,
        max_chars=max_chars,
        docs_per_row_group=docs_per_row_group,
        layout=layout,
        first=first,
        lang_pattern=lang_pattern,
        detect_lang=detect_lang,
        code_language_column=code_language_column,
        detect_code=detect_code,
        chat_template=chat_template,
        chat_template_revision=chat_template_revision,
    )
    return write_corpus(texts, out, ascii=ascii, labels=labels, **source)


def _sample_hub(
    dataset: str,
    *,
    files: str,
    text_column: str,
    revision: str | None,
    documents: int,
    seed: int,
    max_chars: int,
    docs_per_row_group: int,
    layout: str | None,
    first: bool = False,
    lang_pattern: str | None = None,
    detect_lang: bool = False,
    code_language_column: str | None = None,
    detect_code: bool = False,
    chat_template: str | None = None,
    chat_template_revision: str | None = None,
) -> tuple[list[str], dict | None, dict]:
    """Return the documents that :func:`build_from_hub` samples, and their labels.

    Also returns the fields of their :class:`CorpusSource`, except those of the written file.
    """
    _check_options(layout, lang_pattern, detect_lang, code_language_column, detect_code)
    chat, chat_template_revision = _chat(chat_template, chat_template_revision, layout)
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    info = api.dataset_info(dataset, revision=revision)
    license_ = _license(api, dataset, info, revision)
    fs = HfFileSystem()
    code_languages: list[str | None] = []
    column = {"label_column": code_language_column, "labels": code_languages, "chat": chat}
    read = {"text_column": text_column, "max_chars": max_chars, "layout": layout, **column}
    sampled = {"seed": seed, "docs_per_row_group": docs_per_row_group}
    reader = first_documents if first else partial(sample_documents, **sampled)
    take = partial(reader, fs, **read)
    texts, globs = _read_globs(fs, f"{dataset}@{info.sha}", files, documents, take)
    labels, detector = _document_labels(texts, globs, lang_pattern, detect_lang)
    code_detector = _code_labels(labels, texts, code_language_column, code_languages, detect_code)
    source = dict(
        dataset=dataset,
        revision=info.sha,
        files=files,
        text_column=text_column,
        seed=seed,
        max_chars=max_chars,
        docs_per_row_group=docs_per_row_group,
        layout=layout,
        first=first,
        lang_pattern=lang_pattern,
        lang_detector=detector,
        code_language_column=code_language_column,
        code_detector=code_detector,
        chat_template=chat_template,
        chat_template_revision=chat_template_revision,
        license=", ".join(license_) if isinstance(license_, list) else license_,
    )
    return texts, labels or None, source


def _check_options(layout, lang_pattern, detect_lang, code_language_column, detect_code):
    """Raise ``ValueError`` for labelling options of :func:`_sample_hub` that can't go together."""
    if lang_pattern and detect_lang:
        raise ValueError("a corpus's languages come from lang_pattern or detect_lang, not both")
    if code_language_column and detect_code:
        raise ValueError(
            "a corpus's programming languages come from code_language_column or detect_code, "
            "not both"
        )
    if code_language_column and layout:
        raise ValueError("a layout joins rows, so a column can't label its documents")


def _chat(chat_template: str | None, revision: str | None, layout: str | None):
    """Return the chat formatter of ``chat_template`` and its pinned revision, or ``None``s.

    Raises ``ValueError`` with a ``layout``, which joins rows of text, not conversations.
    """
    if not chat_template:
        return None, None
    if layout:
        raise ValueError("a layout joins rows of text, not conversations")
    return chat_formatter(chat_template, revision)


def _document_labels(
    texts: list[str], globs: list[str], lang_pattern: str | None, detect_lang: bool
) -> tuple[dict, str | None]:
    """Return the ``lang`` labels, if the corpus has them, and the detector that gave them."""
    if detect_lang:
        from openinterp import langid

        return {"lang": langid.detect(texts)}, langid.DETECTOR
    return ({"lang": [_glob_lang(lang_pattern, g) for g in globs]} if lang_pattern else {}), None


def _code_labels(
    labels: dict, texts: list[str], column: str | None, exact: list, detect_code: bool
) -> str | None:
    """Add each document's programming language to ``labels``, if the corpus has them.

    With a dataset ``column``, ``exact`` holds its values. With ``detect_code``,
    :mod:`openinterp.codeid` detects the languages. Returns the detector, or ``None``.
    """
    if column:
        labels["code_language"] = exact
    elif detect_code:
        from openinterp import codeid

        labels["code_language"] = codeid.detect(texts)
        return codeid.detector()
    return None


def _read_globs(fs, repo: str, files: str, documents: int, take):
    """Return the documents that ``take(paths, documents=n)`` reads from ``files``' globs.

    Also returns the glob that each came from. Several globs (for example, one per
    language) give an equal share each, and the first globs take the extras.
    """
    texts, origins = [], []
    globs = files.split(",")
    for i, glob in enumerate(globs):
        paths = fs.glob(f"datasets/{repo}/{glob}")
        if not paths:
            raise FileNotFoundError(f"no files match {glob!r} in {repo}")
        share = documents // len(globs) + (i < documents % len(globs))
        texts += take(paths, documents=share)
        origins += [glob] * (len(texts) - len(origins))
    return texts, origins


def _glob_lang(pattern: str | None, glob: str) -> str | None:
    match = re.search(pattern, glob) if pattern else None
    if pattern and match is None:
        raise ValueError(f"{glob!r} names no language under the pattern {pattern!r}")
    if match is None:
        return None
    return FINEWEB2_LANGUAGES.get(match.group(1), match.group(1))


def _license(api, dataset: str, info, revision: str | None):
    """The dataset card's licence (a string or a list), else :data:`UNSTATED_LICENSES`'s."""
    if info.card_data is None and revision is not None:
        # Branches like refs/convert/parquet carry no dataset card: use the default branch's.
        card_data = api.dataset_info(dataset).card_data
    else:
        card_data = info.card_data
    card = card_data.to_dict() if card_data else {}
    # OpenWebMath's card puts the licence inside its single `dataset_info`, which is a mapping.
    # Cards with several configs have a list there, and it holds no licence.
    nested = card.get("dataset_info")
    nested_license = nested.get("license") if isinstance(nested, dict) else None
    return card.get("license") or nested_license or UNSTATED_LICENSES.get(dataset)
