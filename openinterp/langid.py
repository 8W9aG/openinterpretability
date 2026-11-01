"""A document's language, for corpora whose dataset doesn't say (``openinterp corpus
--detect-lang``).

The detector is fastText's ``lid.176`` (compressed, 176 languages), chosen over lingua,
langdetect and langid.py on labelled Wikipedia (``docs/analyses/language-detection.md``):
it was the most accurate at every length and the fastest, and it alone is unsure of
source code. A document is labelled only when the detector's confidence reaches
``MIN_CONFIDENCE``, so code, markup and mixed text mostly stay unlabelled, which the scan
leaves out of the language confound. Detection is per document, never per window: a
window of a few dozen tokens is too short to detect reliably.

The model (CC BY-SA 3.0) is downloaded once, checked against ``MODEL_SHA256`` and cached
in ``$XDG_CACHE_HOME/openinterp`` (else ``~/.cache/openinterp``).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Sequence
from functools import cache
from pathlib import Path

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
# Below this the document is left unlabelled. On labelled Wikipedia it keeps 98.5% of
# documents at 99.9% accuracy, and labels 9% of source files (0.5 would label 45%).
MIN_CONFIDENCE = 0.8
DETECTOR = f"fasttext lid.176.ftz (sha256 {MODEL_SHA256[:12]}), confidence >= {MIN_CONFIDENCE}"


def model_path() -> Path:
    """The cached model, downloaded and checked on first use."""
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    path = base / "openinterp" / Path(MODEL_URL).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL) as response:  # noqa: S310  (fixed https URL)
            data = response.read()
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise ValueError(f"{MODEL_URL} does not match its pinned SHA-256")
        partial = path.with_suffix(".part")
        partial.write_bytes(data)
        partial.replace(path)
    return path


@cache
def _model():
    import fasttext

    return fasttext.load_model(str(model_path()))


def predict(text: str) -> tuple[str, float]:
    """The most likely language of ``text`` (the model's code, e.g. ``"en"``, ``"ceb"``)
    and the detector's confidence in it."""
    # fastText reads one line; ``f.predict`` returns [(probability, label)] without numpy.
    ((p, label),) = _model().f.predict(" ".join(text.split()), 1, 0.0, "strict")
    return label.removeprefix("__label__"), float(p)


def detect(texts: Sequence[str]) -> list[str | None]:
    """Each document's language, or ``None`` below :data:`MIN_CONFIDENCE`."""
    out: list[str | None] = []
    for text in texts:
        label, p = predict(text)
        out.append(label if p >= MIN_CONFIDENCE else None)
    return out
