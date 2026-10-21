"""Build scan corpora from Hugging Face datasets (`openinterp corpus`).

A corpus is a JSONL file of ``{"text": ...}`` documents, plus a ``<name>.source.json``
sidecar (:class:`CorpusSource`) that pins where it came from. Documents are sampled
without downloading whole datasets: Parquet row groups are read over HTTP, so a
500-document sample of fineweb-edu's 10B-token sample costs a few dozen small reads.

Sampling is deterministic given the dataset revision and seed: row groups are visited
in a seeded random order across all files, and a seeded handful of documents is taken
from each, which spreads the corpus over sources and crawl dumps.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from openinterp.core.schemas import CorpusSample, CorpusSource

# What `openinterp audit` scans by default (protocol 0.7): a fixed seed, so every audit
# of a corpus reads the same documents and their scales are comparable.
DEFAULT_SAMPLE_TOKENS = 200_000
DEFAULT_SAMPLE_SEED = 0


@dataclass(frozen=True)
class Preset:
    dataset: str
    files: str  # glob within the dataset repo
    text_column: str = "text"
    revision: str | None = None  # default branch or tag; pinned to a commit on use


PRESETS = {
    # Clean, educational web text (FineWeb filtered by an educational-quality classifier),
    # closer to recent models' pretraining mix than the Pile. ODC-By: attribute excerpts.
    "fineweb-edu": Preset("HuggingFaceFW/fineweb-edu", "sample/10BT/*.parquet"),
    # The first 10k Pile documents; the registry's original corpus.
    "pile-10k": Preset("NeelNanda/pile-10k", "data/*.parquet"),
    # C4 English validation split: the standard web baseline of the SAE literature. The
    # main branch is gzipped JSON, so read the Hub's Parquet conversion. ODC-By.
    "c4": Preset("allenai/c4", "en/partial-validation/*.parquet", revision="refs/convert/parquet"),
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
) -> list[str]:
    """Up to ``documents`` non-empty texts, cut to ``max_chars``, from Parquet files on ``fs``."""
    rng = random.Random(seed)
    groups = [(path, i) for path in sorted(paths) for i in range(_row_groups(fs, path))]
    rng.shuffle(groups)
    # Few row groups (a small dataset): take more from each, so the sample still fills.
    take = max(docs_per_row_group, math.ceil(documents / max(len(groups), 1)))
    out: list[str] = []
    for texts in _read_groups(fs, groups, text_column):
        order = rng.sample(range(len(texts)), len(texts))
        picked = [j for j in order if (texts[j] or "").strip()][:take]  # empties don't count
        for j in sorted(picked):
            out.append(texts[j].strip()[:max_chars])
            if len(out) >= documents:
                return out
    return out


def _row_groups(fs, path: str) -> int:
    with fs.open(path, "rb") as fh:
        return pq.ParquetFile(fh).num_row_groups


def _read_groups(fs, groups, column: str) -> Iterator[list]:
    for path, i in groups:
        with fs.open(path, "rb") as fh:
            yield pq.ParquetFile(fh).read_row_group(i, columns=[column]).column(column).to_pylist()


def jsonl_bytes(texts: Sequence[str]) -> bytes:
    """``texts`` as a JSONL corpus: one ``{"text": ...}`` per line."""
    return "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts).encode()


def sample_tokens(
    tokenizer, texts: Sequence[str], max_tokens: int, seed: int
) -> tuple[list[str], CorpusSample]:
    """A seeded sample of whole documents with at most ``max_tokens`` tokens (see
    :class:`CorpusSample`), in corpus order, and its description."""
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
    sample = [texts[i] for i in sorted(picked)]
    return sample, CorpusSample(
        seed=seed,
        max_tokens=max_tokens,
        documents=len(sample),
        corpus_documents=len(texts),
        tokens=total,
        sha256=hashlib.sha256(jsonl_bytes(sample)).hexdigest(),
    )


def write_corpus(texts: list[str], out: Path, **source) -> CorpusSource:
    """Write the JSONL corpus and its source sidecar; returns the source."""
    out.parent.mkdir(parents=True, exist_ok=True)
    data = jsonl_bytes(texts)
    out.write_bytes(data)
    record = CorpusSource(documents=len(texts), sha256=hashlib.sha256(data).hexdigest(), **source)
    source_path(out).write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record


def load_source(corpus_path: Path, sha256: str) -> CorpusSource | None:
    """The corpus's source sidecar, if it exists and describes these exact bytes."""
    path = source_path(corpus_path)
    if not path.exists():
        return None
    source = CorpusSource.model_validate_json(path.read_text(encoding="utf-8"))
    return source if source.sha256 == sha256 else None


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
) -> CorpusSource:
    """Sample a corpus from a Hugging Face dataset at a pinned revision."""
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    info = api.dataset_info(dataset, revision=revision)
    if info.card_data is None and revision is not None:
        # Branches like refs/convert/parquet carry no dataset card: use the default branch's.
        card_data = api.dataset_info(dataset).card_data
    else:
        card_data = info.card_data
    license_ = card_data.to_dict().get("license") if card_data else None
    fs = HfFileSystem()
    paths = fs.glob(f"datasets/{dataset}@{info.sha}/{files}")
    if not paths:
        raise FileNotFoundError(f"no files match {files!r} in {dataset}@{info.sha}")
    texts = sample_documents(
        fs,
        paths,
        text_column=text_column,
        documents=documents,
        seed=seed,
        max_chars=max_chars,
        docs_per_row_group=docs_per_row_group,
    )
    return write_corpus(
        texts,
        out,
        dataset=dataset,
        revision=info.sha,
        files=files,
        text_column=text_column,
        seed=seed,
        max_chars=max_chars,
        docs_per_row_group=docs_per_row_group,
        license=", ".join(license_) if isinstance(license_, list) else license_,
    )
