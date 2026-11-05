"""Find people's names in a document, for the scan's ``names`` label set.

The label is a model's guess, for experiments only. Nothing removes text because of it.
On a 500-document sample it matched names in 47% of documents. Most are public and
fictional people, so a match does not say that the text is personal data.

The detector is spaCy's ``en_core_web_trf``, the best of spaCy's four English models. On
CoNLL-2003's person names they had F1 0.67, 0.71, 0.78 and 0.89. We also compared
them on a sample of web text, code and other languages. There the smaller models also
called code and non-English phrases names. They matched 77%, 71% and 62% of documents.
This model is the slowest, at about 180 ms a document on CPU. So each document's result
stays in a cache on disk. It is an English model, so it reads other languages less well.

The model (MIT) is about 460 MB. It is downloaded once, checked against ``MODEL_SHA256``
and cached in ``$XDG_CACHE_HOME/openinterp`` (else ``~/.cache/openinterp``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from functools import cache
from pathlib import Path

MODEL = "en_core_web_trf-3.8.0"
MODEL_URL = (
    f"https://github.com/explosion/spacy-models/releases/download/{MODEL}/{MODEL}-py3-none-any.whl"
)
MODEL_SHA256 = "272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5"
DETECTOR = f"spaCy {MODEL} (sha256 {MODEL_SHA256[:12]})"
PERSON = "person"  # the label of a match
_PIPES = ("transformer", "ner")  # the others are slower, and names don't need them


def _cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "openinterp"


def _download(wheel: Path) -> None:
    """Write the model's wheel to ``wheel``, or raise if it does not match its pinned SHA-256."""
    digest = hashlib.sha256()
    partial = wheel.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        partial.unlink()
        raise ValueError(f"{MODEL_URL} does not match its pinned SHA-256")
    partial.replace(wheel)


def model_path() -> Path:
    """The unpacked model, downloaded and checked on first use."""
    root = _cache_dir() / MODEL
    path = root / "en_core_web_trf" / MODEL
    if not path.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        wheel = root.with_suffix(".whl")
        _download(wheel)
        unpacking = root.with_suffix(".unpacking")
        shutil.rmtree(unpacking, ignore_errors=True)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(unpacking)
        unpacking.replace(root)
        wheel.unlink()
    return path


@cache
def _model():
    import spacy

    nlp = spacy.load(model_path())
    return spacy.load(model_path(), exclude=[p for p in nlp.pipe_names if p not in _PIPES])


def person_matches(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return ``(start, end, PERSON)`` of each name in ``text``, sorted.

    The result stays on disk by the text's SHA-256, so a later scan of a corpus does not
    run the model again.
    """
    key = hashlib.sha256(text.encode()).hexdigest()
    path = _cache_dir() / "names" / MODEL_SHA256[:12] / f"{key}.json"
    if path.exists():
        return tuple((start, end, PERSON) for start, end in json.loads(path.read_text()))
    found = sorted(
        (ent.start_char, ent.end_char) for ent in _model()(text).ents if ent.label_ == "PERSON"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.part")  # scans can run side by side
    partial.write_text(json.dumps(found))
    partial.replace(path)
    return tuple((start, end, PERSON) for start, end in found)
